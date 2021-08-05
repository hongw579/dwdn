    def mean_x(self, img):
        Itot = torch.sum(img, (3,2))
        x_idx = torch.arange(1, img.shape[-2]+1).to(self.device)
        y_sum = torch.sum(img, (3))
        mean_x = torch.sum(x_idx*y_sum, (-1))/Itot
        return mean_x

    def mean_y(self, img):
        Itot = torch.sum(img, (3,2))
        y_idx = torch.arange(1, img.shape[-1]+1).to(self.device)
        x_sum = torch.sum(img, (2))
        mean_y = torch.sum(y_idx*x_sum, (-1))/Itot
        return mean_y

    def Mxx(self, img):
        mean_x = self.mean_x(img).unsqueeze(-1)
        x_idx = torch.arange(1, img.shape[-2]+1).to(self.device)
        x_idx = x_idx.repeat(img.shape[0], img.shape[1], 1)
        y_sum = torch.sum(img, (3))
        M_xx = torch.sum(((x_idx-mean_x)**2)*y_sum, (-1))
        return M_xx

    def Myy(self, img):
        mean_y = self.mean_y(img).unsqueeze(-1)
        y_idx = torch.arange(1, img.shape[-1]+1).to(self.device)
        y_idx = y_idx.repeat(img.shape[0], img.shape[1], 1)
        x_sum = torch.sum(img, (2))
        M_yy = torch.sum(((y_idx-mean_y)**2)*x_sum, (-1))
        return M_yy

    def Mxy(self, img):
        mean_x = self.mean_x(img).unsqueeze(-1).unsqueeze(-1)
        mean_y = self.mean_y(img).unsqueeze(-1).unsqueeze(-1)
        x_idx = torch.arange(1, img.shape[-2]+1).to(self.device)
        x_idx = x_idx.repeat(img.shape[0], img.shape[1], 1).unsqueeze(-1)
        y_idx = torch.arange(1, img.shape[-1]+1).to(self.device)
        y_idx = y_idx.repeat(img.shape[0], img.shape[1], 1).unsqueeze(-2)
        M_xy = torch.sum(((x_idx-mean_x)*(y_idx-mean_y)*img), (-1, -2))
        return M_xy
