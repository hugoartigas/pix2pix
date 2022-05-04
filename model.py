from re import X
import torch
import torch.nn as nn
import utils

class Generator(nn.Module):
    """
    The Generator with U-net architecture
    """
    def __init__(self,n_channel = 3):

        super(Generator, self).__init__()
        self.n_channel = n_channel
        ## Defining the down sample layers
        self.conv1  = nn.Sequential(
            nn.Conv2d(self.n_channel,self.n_channel, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_channel,self.n_channel*2, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channel*2)).to(utils.device)
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.n_channel*2,self.n_channel*4, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channel*4),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.n_channel*4,self.n_channel*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channel*8),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.n_channel*8,self.n_channel*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channel*8),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.n_channel*8,self.n_channel*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channel*8),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv7 = nn.Sequential(
            nn.Conv2d(self.n_channel*8,self.n_channel*8, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_channel*8),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv8 = nn.Sequential(
            nn.Conv2d(self.n_channel*8,self.n_channel*8, kernel_size = 2, stride = 1, padding=0, bias=False)).to(utils.device)
        self.batchnorm = nn.Sequential(
            nn.BatchNorm2d(self.n_channel*8),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.batchnorm = nn.Sequential(
            nn.BatchNorm2d(self.n_channel*8),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.leaky = nn.LeakyReLU(negative_slope=0.3).to(utils.device)
        # Defining the Dropout function
        self.dropout = nn.Dropout(0.5).to(utils.device)
        # Defing the upsample layers
        self.upconv1 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(self.n_channel*8, self.n_channel*8,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel*8)).to(utils.device)
        self.upconv2 = nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(2*self.n_channel*8, self.n_channel*8, kernel_size=4, stride=2,padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel*8)).to(utils.device)
        self.upconv3 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channel*8, self.n_channel*8,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel*8)).to(utils.device)
        self.upconv4 = nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(2*self.n_channel*8, self.n_channel*8, kernel_size=4, stride=2,padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel*8)).to(utils.device)
        self.upconv5 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channel*8, self.n_channel*4,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel*4)).to(utils.device)
        self.upconv6 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channel*4, self.n_channel*2, kernel_size=4, stride=2,padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel*2)).to(utils.device)
        self.upconv7 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channel*2, self.n_channel,kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.n_channel)).to(utils.device)
        self.upconv8 = nn.Sequential( nn.ReLU(inplace=True),nn.ConvTranspose2d(2*self.n_channel, self.n_channel, kernel_size=4, stride=2,padding=1, bias=True),
            nn.Tanh()).to(utils.device)

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
        outup1 = self.upconv1(out8)
        outp1 = torch.cat((self.dropout(outup1),out7),1)
        outup2 = self.upconv2(outp1)
        outp2 = torch.cat((self.dropout(outup2),out6),1)
        outup3 = self.upconv3(outp2)
        outp3 = torch.cat((self.dropout(outup3),out5),1)
        outup4 = self.upconv4(outp3)
        outp4 = torch.cat((outup4,out4),1)
        outup5 = self.upconv5(outp4)
        outp5 = torch.cat((outup5,out3),1)
        outup6 = self.upconv6(outp5)
        outp6 = torch.cat((outup6,out2),1)
        outup7 = self.upconv7(outp6)
        outp7 = torch.cat((outup7,out1),1)
        outup8 = self.upconv8(outp7)
        return outup8

class Discriminator(nn.Module):
    """
    The Discriminator with the patchGAN architecture
    """
    def __init__(self,n_channel = 3):

        super(Discriminator, self).__init__()
        self.n_channel = n_channel
        self.conv1  = nn.Sequential(
            nn.Conv2d(self.n_channel,64, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.padd = torch.nn.ZeroPad2d(1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=4, stride = 1, bias=False),
            nn.BatchNorm2d(512),nn.LeakyReLU(negative_slope=0.3)).to(utils.device)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,1, kernel_size=4, stride = 1, bias=False)).to(utils.device)


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
        self.generator = Generator().to(utils.device)
        self.discriminator = Discriminator().to(utils.device)

    def forward(self,x):
        gen = self.generator(x)
        discr = self.discriminator(x)
        return gen,discr

