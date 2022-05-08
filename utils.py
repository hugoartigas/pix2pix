from barbar import Bar
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss, L1Loss

# set paths
images_path = os.path.join(os.getcwd(), 'images') 

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set data transformation dictionary for training, validation
data_transforms = {
    'train': T.Compose([
        T.ToTensor(),
        T.Resize((286,286)),
        T.RandomCrop((256,256))
    ]),
    'val': T.Compose([
        T.ToTensor(),
        T.Resize((256,256))
    ])
}


def get_train_test_image_names(set, images_path = images_path):
    """
    Description
    -------------
    List names of train and test videos
    Parameters
    -------------
    dataset_name   : name of dataset
    labels_path    : path to datasets 
    Returns
    -------------
    (train_names, val_names) , each of the tuple elements is a list of strings corresponding to images names
    """
    images_names = {}
    images_path = os.path.join(images_path, set) 
    
    for mode in ['train','val']:
        # list image names and 
        images_names[mode] = os.listdir(os.path.join(images_path, mode))
        images_names[mode].sort()

    return images_names


class ImageDataset(Dataset):
    """Instance of the image dataset."""

    def __init__(self, dataset_name, transform = None, mode = 'train'):
        """
        Description
        -------------
        Creates dataset class for the training set.
        Parameters
        -------------
        dataset_name            : name of dataset
        transform               : transforms to be applied to the frame (eg. data augmentation)
        mode                    : string, either 'train' or 'val'
        Returns
        -------------
        Torch Dataset for the training set
        """
        # set class parameters
        self.images_folder_path = images_path + '/' + dataset_name + '/' + mode
        self.transform = transform
        # set image names
        self.images_names = get_train_test_image_names(dataset_name)[mode]

    def __len__(self):
        # return length of the list of image names
        return len(self.images_names)

    def __getitem__(self, index):
        # recover image path
        image_path =  self.images_folder_path + '/' + self.images_names[index] 
        # read image
        image = cv2.imread(image_path)
        w = image.shape[1]
        w = w // 2
        # recover input and real image
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]
        # apply transformation
        if self.transform:
            input_image = self.transform(input_image)
            real_image = self.transform(real_image)
        # rescale between -1 and 1 (linear mapping [0,1] to [-1,1])
        input_image = (input_image / 0.5) -  1
        real_image = (real_image / 0.5) - 1

        return input_image, real_image

def generate_images(model, input, real):
    """
    Description
    -------------
    Plot input, real image and model predictions side by side
    Parameters
    -------------
    input       : input image
    model       : Pix2Pix model
    """
    prediction = model.generator(input.to(device))

    # create figure
    plt.figure(figsize=(15, 15))

    # recover image of each batch of size 1
    display_list = [input[0], real[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def train(model, n_epochs, dataloader):
    """
    Training of the Pix2Pix model by firstly trainin the generator and then training the discriminator
    """

    """
    Description
    -------------
    Train the Pix2Pix model.
    Parameters
    -------------
    model                   : model to train
    n_epochs                : number of epochs to train the model on
    dataloader              : dataloader to use for training
    Returns
    -------------
    Torch Dataset for the training set
    """
    # set weight of l1 loss within generator loss
    lbd = 100

    def generator_loss(disc_generated_output, gen_output, real):
        gan_loss = torch.nn.BCEWithLogitsLoss()(disc_generated_output, Variable(torch.ones_like(disc_generated_output)))
        l1_loss = torch.nn.L1Loss()(real, gen_output)
        total_gen_loss = gan_loss + (lbd * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = torch.nn.BCEWithLogitsLoss()(disc_real_output, Variable(torch.ones_like(disc_real_output)))
        generated_loss = torch.nn.BCEWithLogitsLoss()(disc_generated_output, Variable(torch.zeros_like(disc_generated_output)))
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    # optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-07)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-07)

    since = time.time()

    # go in train mode
    model.train()

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        for input, real in Bar(dataloader):
            # send tensors to device
            input = input.to(device)
            real = real.to(device)

            # we first update discriminator
            model.discriminator.set_requires_grad(True)
            optimizer_D.zero_grad()
            gen_output = model.generator(input)
            disc_real_output = model.discriminator(input, real)
            disc_generated_output = model.discriminator(input, gen_output)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            disc_loss.backward()
            optimizer_D.step()

            # and then generator
            model.discriminator.set_requires_grad(False)
            optimizer_G.zero_grad()
            gen_output = model.generator(input)
            disc_real_output = model.discriminator(input, real)
            disc_generated_output = model.discriminator(input, gen_output)
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, real)
            gen_total_loss.backward()
            optimizer_G.step()
            


        print(f'gen_total_loss: {gen_total_loss:.4f}, gen_gan_loss: {gen_gan_loss:.4f}, gen_l1_loss: {gen_l1_loss:.4f}, disc_loss: {disc_loss:.4f}.')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

