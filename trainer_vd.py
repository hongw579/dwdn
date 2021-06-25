import os
import decimal
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm

import utils_deblur
import torch.nn.functional as F

class Trainer_VD:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args

        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp
        self.error_last = 1e8

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def clip_gradient(self, optimizer, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        print("Image Deblur Training")
        #self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        print('Epoch: ', epoch)
        for batch, (blur, sharp, kernel) in enumerate(self.loader_train):

            blur = blur.to(self.device)
            sharp = sharp.to(self.device)

            self.optimizer.zero_grad()

            deblur = self.model(blur, kernel)
            self.n_levels = 2
            self.scale = 0.5
            loss = 0
            for level in range(self.n_levels):
                scale = self.scale ** (self.n_levels - level - 1)
                n, c, h, w = sharp.shape
                hi = int(round(h * scale))
                wi = int(round(w * scale))
                sharp_level = F.interpolate(sharp, (hi, wi), mode='bilinear')
                loss = loss + self.loss(deblur[level], sharp_level)

            self.ckp.report_log(loss.item())
            loss.backward()
            self.clip_gradient(self.optimizer, self.args.grad_clip)
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.loss.display_loss(batch)))
        if epoch % 10 == 0:
           self.model.save(self.ckp.dir, epoch)

        self.scheduler.step()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch
        self.model.eval()
        self.ckp.start_log(train=False)
        image = []
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (blur, sharp, kernel) in enumerate(tqdm_test):

                blur = blur.to(self.device)

                deblur = self.model(blur, kernel) #deblur[-1].shape=1,1,32,32, len(deblur)=2; deblur[-1].shape=1,1,16,16

                image.append(deblur[-1])
                if (idx_img+1)%3 == 0:
                    image_3c = torch.cat(image, 1)
                    image = []

                    if self.args.save_images and (idx_img+1)/3<=20:
                        deblur = utils_deblur.postprocess(image_3c, rgb_range=self.args.rgb_range)
                        save_list = [deblur[0]]
                        self.ckp.save_images(str(int((idx_img+1)/3)), save_list)

            self.ckp.end_log(len(self.loader_test), train=False)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
