from barbar import Bar
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
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
    
    # losses
    adversarial_loss = BCEWithLogitsLoss().to(device)
    criterionL1 = L1Loss().to(device)
    l = 100 # L1 regularization

    # optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=0.0002,betas=(0.5, 0.999),eps=1e-07)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002,betas=(0.5, 0.999),eps=1e-07)

    since = time.time()

    # go in train mode
    model.train()

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_gen_loss = 0.0
        running_discr_loss = 0.0
        # iterate over data
        for inputs, reals in Bar(dataloader):
            inputs = inputs.to(device)
            reals = reals.to(device)

            # zero the parameter gradients
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
        
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = model.generator(inputs).to(device)
            # Loss measures generator's ability to fool the discriminator
            discrs = model.discriminator(gen_imgs,inputs).to(device)
            valid = torch.ones_like(discrs).to(device)
            fake = torch.zeros_like(discrs).to(device)
            g_loss = adversarial_loss(discrs, valid) + l*criterionL1(gen_imgs,reals)

            g_loss.backward()
            optimizer_G.step()
            running_gen_loss += g_loss.item()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(model.discriminator(reals,inputs), valid)
            fake_loss = adversarial_loss(model.discriminator(gen_imgs.detach(),inputs), fake)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_D.step()
            running_discr_loss += d_loss.item()
            # statistics
            running_loss += d_loss.item()

    

        print(f'Loss: {running_loss:.4f}, generator loss: {running_gen_loss:.4f}, discriminator loss: {running_discr_loss:.4f}.')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
