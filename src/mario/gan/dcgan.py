"""DCGAN intended for Mario.

Adapted from
https://github.com/icaros-usc/MarioGAN-LSI/blob/master/util/models/dcgan.py

Refer to the PyTorch DCGAN tutorial for more info
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
from pathlib import Path

import fire
import gin
import numpy as np
import torch
import torch.nn.parallel
from torch import nn

from src.device import DEVICE

from ..level import MarioLevel


# Formerly known as DCGAN_G.
@gin.configurable
class MarioGenerator(nn.Module):

    def __init__(self,
                 isize: int = gin.REQUIRED,
                 lvl_width: int = gin.REQUIRED,
                 lvl_height: int = gin.REQUIRED,
                 nz: int = gin.REQUIRED,
                 nc: int = gin.REQUIRED,
                 ngf: int = gin.REQUIRED,
                 ngpu: int = gin.REQUIRED,
                 n_extra_layers: int = gin.REQUIRED,
                 model_file: str = gin.REQUIRED):
        super().__init__()

        if isinstance(model_file, tuple) and not model_file[1]:
            # Get full path from first entry of tuple.
            self.model_file = model_file[0]
        else:
            # Set model file relative to the data directory.
            self.model_file = Path(__file__).parent / "data" / model_file

        self.lvl_width = lvl_width
        self.lvl_height = lvl_height

        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                'pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.ReLU())  #nn.Softmax(1))    #Was TANH nn.Tanh())#
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        #print (output[0,:,0,0])
        #exit()
        return output

    def levels_from_latent(self, latent: np.ndarray) -> np.ndarray:
        """Generates Mario levels from latent vectors.

        Args:
            latent: (n, nz) array of latent vectors.
        Returns:
            Each level is a lvl_height x lvl_width array of integers where each
            integer is the type of object, so the output of this method is an
            array of levels of shape (n, lvl_height, lvl_width).
        """
        # Handle no_grad here since we expect everything to be numpy arrays.
        with torch.no_grad():
            latent = torch.as_tensor(np.asarray(latent),
                                     dtype=torch.float,
                                     device=DEVICE)[:, :, None, None]
            lvls = self(latent)
            cropped_lvls = lvls[:, :, :self.lvl_height, :self.lvl_width]
            # Convert from one-hot encoding to ints.
            int_lvls = torch.argmax(cropped_lvls, dim=1)
            return int_lvls.cpu().detach().numpy()

    def load_from_saved_weights(self):
        self.load_state_dict(torch.load(self.model_file, map_location=DEVICE))
        return self


# Formerly known as DCGAN_D.
class MarioDiscriminator(nn.Module):

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super().__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:conv:{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:relu:{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


###############################################################################
class DCGAN_D_nobn(nn.Module):

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial:conv:{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:relu:{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


class DCGAN_G_nobn(nn.Module):

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:relu'.format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                'pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc), nn.Softmax())  #Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


def generator_demo():
    generator = MarioGenerator(
        isize=64,
        lvl_width=56,
        lvl_height=16,
        nz=32,
        nc=17,
        ngf=64,
        ngpu=1,
        n_extra_layers=0,
        model_file="netG_epoch_4999_7684.pth",
    ).load_from_saved_weights().to(DEVICE).eval()

    levels = list(
        map(MarioLevel,
            generator.levels_from_latent(np.random.standard_normal((2, 32)))))

    for i, level in enumerate(levels, 1):
        print(f"==> LEVEL {i} <==")
        print(level)


if __name__ == "__main__":
    fire.Fire(generator_demo)
