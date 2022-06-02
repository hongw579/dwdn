import os
import numpy as np
import torch
import median_pool
import matplotlib.pyplot as plt
import math
'''
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
'''

sigma=5.
gaussian = [[math.exp(-(i**2+j**2)/(2*sigma**2)) for i in range(-51,51)] for j in range(-51,51)]
gaussian_tensor = torch.tensor(gaussian).unsqueeze(-1).to('cuda')

# --------------------------------
# --------------------------------
def get_uperleft_denominator(img, kernel):
    #ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    nsr = wiener_filter_para(img)
    ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    denominator = inv_fft_kernel_est(ker_f, nsr )#
    denominator *= gaussian_tensor
    img1 = img.cuda()
    numerator = torch.rfft(img1, 3, onesided=False)
    deblur = deconv(denominator, numerator)
    return deblur

# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur):
    median_filter = median_pool.MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[2])
    mean_n = torch.sum(diff, (2,3)).view(-1,1,1,1)/num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2,3))/(num-1)
    mean_input = torch.sum(_input_blur, (2,3)).view(-1,1,1,1)/num
    var_s2 = (torch.sum((_input_blur-mean_input)*(_input_blur-mean_input), (2,3))/(num-1))**(0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1,1,1,1)
    return NSR

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] \
                      + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] + NSR
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / (inv_denominator+1e-12)
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / (inv_denominator+1e-12)
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
                            - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
                            + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur = torch.irfft(deblur_f, 3, onesided=False)
    return deblur

# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size):
    psf = torch.zeros(size).cuda()
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    otf = torch.rfft(psf, 3, onesided=False)
    return otf


def postprocess(*images, rgb_range):
    #print(type(images))
    def _postprocess(img):
        print('img shape in postprocess',img.shape)
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(images[0])]
    #return [_postprocess(img) for img in images]

def plotting(idx, kernel, blur, deblur, true, filename='./results/', norm=False):
    kernel = kernel[0].clamp(0, 1).data.numpy()
    blur = blur[0].clamp(0, 1).data.cpu().numpy()
    deblur = deblur[0].clamp(0, 1).data.cpu().numpy()
    true = true[0].clamp(0, 1).data.cpu().numpy()

    kernel = np.transpose(kernel, (1, 2, 0))/kernel.max()
    blur = np.transpose(blur, (1, 2, 0))#/blur.max()
    deblur = np.transpose(deblur, (1, 2, 0))#/deblur.max()
    true = np.transpose(true, (1, 2, 0))#/true.max()

    #print(kernel.shape, blur.shape, deblur.shape, true.shape)
    #print(kernel.min(), kernel.max(), blur.min(), blur.max(), deblur.min(), deblur.max(), true.min(), true.max())

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        print('Save Path : {}'.format('./results'))

    plt.subplot(1,4,1)
    plt.imshow(kernel, interpolation='nearest')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(blur, interpolation='nearest')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow(deblur, interpolation='nearest')
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.imshow(true, interpolation='nearest')
    plt.axis('off')
    plt.savefig('{}{}.png'.format(filename, str(idx)))
