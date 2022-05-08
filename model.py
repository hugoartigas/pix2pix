import torch
import torch.nn as nn
import torch.nn.functional as 

class Generator(nn.Module):
    """ Generator of the Pix2Pix model. """

    def __init__(self):
        """
        Description
        -------------
        Initialize generator model
        """
        super(Generator, self).__init__()

        # instantiate downsampling layers
        self.downsample_layers = [self.__create_downsample_layer(3, 64, apply_batchnorm=False),
                                self.__create_downsample_layer(64, 128),
                                self.__create_downsample_layer(128, 256),
                                self.__create_downsample_layer(256, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512)]

        # instantiate upsampling layers
        self.upsample_layers = [self.__create_upsample_layer(512, 512, apply_dropout=True),
                                self.__create_upsample_layer(512, 512, apply_dropout=True),
                                self.__create_upsample_layer(512, 512, apply_dropout=True),
                                self.__create_upsample_layer(512, 512),
                                self.__create_upsample_layer(512, 256),
                                self.__create_upsample_layer(256, 128),
                                self.__create_upsample_layer(128, 64)]
        
        # create last layer
        self.last = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=0, bias=False)
        nn.init.normal_(self.last, mean=0.0, std=0.02)

    def __create_downsample_layer(self, in_channels, out_channels, kernel_size=4, apply_batchnorm = True):
        """
        Description
        -------------
        Creates downsample layer with convolution, batchnorm and activation
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, bias=False)
        nn.init.normal_(conv_layer, mean=0.0, std=0.02)
        layers.append(conv_layer)
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)
        
    def __create_upsample_layer(self, in_channels, out_channels, kernel_size=4, apply_dropout = False):
        """
        Description
        -------------
        Creates upsample layer with deconvolution, batchnorm, dropout and activation
        """
        layers = []
        deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=0, bias=False)
        nn.init.normal_(deconv_layer, mean=0.0, std=0.02)
        layers.append(deconv_layer)
        layers.append(nn.BatchNorm2d(out_channels))
        if apply_dropout:
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    

    def forward(self, x):
        """
        Description
        -------------
        Forward pass
        Parameters
        -------------
        x                : tensor of shape (batch_size, c, w, h)
        """
        
        # downsampling phase
        skips = []
        for down in self.downsample_layers:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # upsampling while concatenating skip filter maps
        for up, skip in zip(self.upsample_layers, skips):
            x = up(x)
            x = torch.cat((x, skip), 1)

        # apply last layer
        x = self.last(x)
        x = nn.Tanh(x)
        
        return x