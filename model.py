import torch
import torch.nn as nn

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
        self.downsample_layers = nn.ModuleList([self.__create_downsample_layer(3, 64, apply_batchnorm=False),
                                self.__create_downsample_layer(64, 128),
                                self.__create_downsample_layer(128, 256),
                                self.__create_downsample_layer(256, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512, apply_batchnorm=False)]) # to modify

        # instantiate upsampling layers
        self.upsample_layers = nn.ModuleList([self.__create_upsample_layer(512, 512, apply_dropout=True),
                                self.__create_upsample_layer(1024, 512, apply_dropout=True),
                                self.__create_upsample_layer(1024, 512, apply_dropout=True),
                                self.__create_upsample_layer(1024, 512),
                                self.__create_upsample_layer(1024, 256),
                                self.__create_upsample_layer(512, 128),
                                self.__create_upsample_layer(256, 64)])
        
        # create last layer
        self.last = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        nn.init.normal_(self.last.weight, mean=0.0, std=0.02)

    def __create_downsample_layer(self, in_channels, out_channels, kernel_size=4, padding = 1, apply_batchnorm = True):
        """
        Description
        -------------
        Creates downsample layer with convolution, batchnorm and activation
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=False)
        nn.init.normal_(conv_layer.weight, mean=0.0, std=0.02)
        layers.append(conv_layer)
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)
        
    def __create_upsample_layer(self, in_channels, out_channels, kernel_size=4, padding=1, apply_dropout = False):
        """
        Description
        -------------
        Creates upsample layer with deconvolution, batchnorm, dropout and activation
        """
        layers = []
        deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=False)
        nn.init.normal_(deconv_layer.weight, mean=0.0, std=0.02)
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
            #print('down output shape : ', x.shape)

        skips = reversed(skips[:-1])

        # upsampling while concatenating skip filter maps
        for up, skip in zip(self.upsample_layers, skips):
            x = up(x)
            x = torch.cat((x, skip), 1)
            #print('up output shape : ', x.shape)

        # apply last layer
        x = self.last(x)
        #print('last output shape : ', x.shape)
        x = nn.Tanh()(x)
        
        return x

class Discriminator(nn.Module):
    """ Discriminator of the Pix2Pix model. """

    def __init__(self):
        """
        Description
        -------------
        Initialize discriminator model
        """
        super(Discriminator, self).__init__()

        # instantiate downsampling layers
        self.down1 = self.__create_downsample_layer(6, 64, 4, apply_batchnorm = False)
        self.down2 = self.__create_downsample_layer(64, 128, 4)
        self.down3 = self.__create_downsample_layer(128, 256, 4)

        self.conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, bias=False)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

        self.batchnorm = nn.BatchNorm2d(512)

        self.last = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)
        nn.init.normal_(self.last.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.last.bias, 0.0)


    def __create_downsample_layer(self, in_channels, out_channels, kernel_size=4, padding = 1, apply_batchnorm = True):
        """
        Description
        -------------
        Creates downsample layer with convolution, batchnorm and activation
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=False)
        nn.init.normal_(conv_layer.weight, mean=0.0, std=0.02)
        layers.append(conv_layer)
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def set_requires_grad(self, bool):
        for param in self.parameters():
            param.requires_grad = bool


    def forward(self, input, real):
        """
        Description
        -------------
        Forward pass
        Parameters
        -------------
        input               : input image, tensor of shape (batch_size, c, w, h)
        real                : real target image, tensor of shape (batch_size, c, w, h)
        """
        
        x = torch.cat([input, real], 1) # concatenate along channels
        #print('output dim : ', x.shape)
        x = self.down1(x)
        #print('output dim : ', x.shape)
        x = self.down2(x)
        #print('output dim : ', x.shape)
        x = self.down3(x)
        #print('output dim : ', x.shape)
        x = nn.ZeroPad2d(1)(x)
        #print('output dim : ', x.shape)
        x = self.conv(x)
        #print('output dim : ', x.shape)
        x = self.batchnorm(x)
        #print('output dim : ', x.shape)
        x = nn.LeakyReLU()(x)
        #print('output dim : ', x.shape)
        x = nn.ZeroPad2d(1)(x)
        #print('output dim : ', x.shape)
        x = self.last(x)
        #print('output dim : ', x.shape)
        return x

class Pix2Pix(nn.Module):
    """
    The Pix2Pix archiecture with U-Net generator and patch GAN disciminator
    """
    def __init__(self):
        super(Pix2Pix, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()