import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    The Generator with U-net architecture
    """
    def __init__(self,n_channels = 3):

        super(Generator, self).__init__()
        self.n_channels = n_channels
        ## Defining the down sample layers
        self.conv1  = nn.Sequential(
            nn.Conv2d(self.n_channels,self.n_channels, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_channels,self.n_channels*2, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channels*2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.n_channels*2,self.n_channels*4, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channels*4),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.n_channels*4,self.n_channels*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channels*8),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.n_channels*8,self.n_channels*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channels*8),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.n_channels*8,self.n_channels*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channels*8),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv7 = nn.Sequential(
            nn.Conv2d(self.n_channels*8,self.n_channels*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channels*8),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv8 = nn.Sequential(
            nn.Conv2d(self.n_channels*8,self.n_channels*8, kernel_size = 2, stride = 1, padding=0, bias=False))
        self.batchnorm = nn.Sequential(
            nn.BatchNorm2d(self.n_channels*8),
            nn.LeakyReLU(negative_slope=0.3))
        self.batchnorm = nn.Sequential(
            nn.BatchNorm2d(self.n_channels*8),
            nn.LeakyReLU(negative_slope=0.3))
        self.leaky = nn.LeakyReLU(negative_slope=0.3)
        # Defining the Dropout function
        self.dropout = nn.Dropout(0.5)
        # Defing the upsample layers
        self.upconv1 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(self.n_channels*8, self.n_channels*8,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels*8))
        self.upconv2 = nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(2*self.n_channels*8, self.n_channels*8, kernel_size=4, stride=2,padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels*8))
        self.upconv3 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channels*8, self.n_channels*8,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels*8))
        self.upconv4 = nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(2*self.n_channels*8, self.n_channels*8, kernel_size=4, stride=2,padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels*8))
        self.upconv5 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channels*8, self.n_channels*4,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels*4))
        self.upconv6 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channels*4, self.n_channels*2, kernel_size=4, stride=2,padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels*2))
        self.upconv7 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channels*2, self.n_channels,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channels))
        self.upconv8 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channels, self.n_channels, kernel_size=4, stride=2,padding=1, bias=True),
            nn.Tanh())

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        out7 = self.conv7(out6)
        out8 = self.conv8(out7)
        if x.shape[0] == 1:
            out8 = self.leaky(out8)
        else:
            out8 = self.batchnorm(out8)
        output1 = self.upconv1(out8)
        outp1 = torch.cat((self.dropout(output1),out7),1)
        output2 = self.upconv2(outp1)
        outp2 = torch.cat((self.dropout(output2),out6),1)
        output3 = self.upconv3(outp2)
        outp3 = torch.cat((self.dropout(output3),out5),1)
        output4 = self.upconv4(outp3)
        outp4 = torch.cat((output4,out4),1)
        output5 = self.upconv5(outp4)
        outp5 = torch.cat((output5,out3),1)
        output6 = self.upconv6(outp5)
        outp6 = torch.cat((output6,out2),1)
        output7 = self.upconv7(outp6)
        outp7 = torch.cat((output7,out1),1)
        output8 = self.upconv8(outp7)
        return output8

class Discriminator(nn.Module):
    """
    The Discriminator with the patchGAN architecture
    """
    def __init__(self,n_channels = 3):

        super(Discriminator, self).__init__()
        self.n_channels = n_channels
        self.conv1  = nn.Sequential(
            nn.Conv2d(self.n_channels,64, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.3))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.3))
        self.padd = torch.nn.ZeroPad2d(1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=4, stride = 1, bias=False),
            nn.BatchNorm2d(512),nn.LeakyReLU(negative_slope=0.3))
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,1, kernel_size=4, stride = 1, bias=False))


    def forward(self,x,y):
        out = torch.cat((x,y),1)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out3 = self.padd(out3)
        out4 = self.conv4(out3)
        out5 = self.padd(out4)
        out6 = self.conv5(out5)
        return out6

class Pix2Pix(nn.Module):
    """
    The Pix2Pix archiecture with U-Net generator and patch GAN disciminator
    """
    def __init__(self):
        super(Pix2Pix, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self,x):
        gen = self.generator(x)
        discr = self.discriminator(x)
        return gen,discr

