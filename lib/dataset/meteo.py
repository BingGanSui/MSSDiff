import torch

class MeteoLoader:
    MinMaxPara = [240., 310., 0., 65.] # [T2m min, T2m max, Prec min, Prec max]
    def __init__(self, path, mode, extract_tag=False):
        import glob
        self.files = glob.glob(('../../' if extract_tag else '')+f'data/{path}/*.pt')
        self.files = sorted(self.files)
        if path == 'demo':
            self.files = self.files
        elif mode == 'train':
            self.files = self.files[:int(len(self.files)*0.78)]
        elif mode == 'val':
            self.files = self.files[int(len(self.files)*0.78):int(len(self.files)*0.8)]
        elif mode == 'test':
            self.files = self.files[int(len(self.files)*0.8):]
        else:
            raise KeyError('[MetaPro]Unknown Mode.')

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def __norm__(self, x):
        """
        :param x: [10, H, W] torch.Tensor
        :return:
        """
        x[:5,:,:] = (x[:5,:,:]-self.MinMaxPara[0])/(self.MinMaxPara[1]-self.MinMaxPara[0])
        # x[5:,:,:] = torch.log(x[5:,:,:]+1)
        x[5:, :, :] = (x[5:, :, :] - self.MinMaxPara[2]) / (self.MinMaxPara[3] - self.MinMaxPara[2])
        return x

    @torch.no_grad()
    def denorm(self, x):
        x[:5,:,:] = x[:5,:,:]*(self.MinMaxPara[1]-self.MinMaxPara[0])+self.MinMaxPara[0]
        x[5:,:,:] = x[5:,:,:]*(self.MinMaxPara[3]-self.MinMaxPara[2])+self.MinMaxPara[2]
        # x[5:,:,:] = torch.exp(x[5:,:,:])-1
        return x

    def __getitem__(self, idx):
        dic = torch.load(self.files[idx], weights_only=True)
        new_dic = {
            'image_lr' : self.__norm__(dic['LR']),
            'image' : self.__norm__(dic['HR']),
            'meta' : self.files[idx].split('/')[-1].replace('.pt', ''),
        }
        return new_dic