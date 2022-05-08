import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pickle
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss, L1Loss

# set paths
images_path = os.path.join(os.getcwd(), 'images') 
weights_path = os.path.join(os.getcwd(), 'weights') 
history_path = os.path.join(os.getcwd(), 'history') 
# create folders if they do not already exist
if not os.path.exists(images_path): os.makedirs(images_path)
if not os.path.exists(weights_path): os.makedirs(weights_path)
if not os.path.exists(history_path): os.makedirs(history_path)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set data transformation dictionary for training, validation
data_transforms = {
    'train': T.Compose([
        # to complete, for now it does not seem to work very well so I commented it
        T.Resize((286,286)),
        T.RandomCrop((256,256))
    ]),
    'val': T.Compose([
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
        image = Image.open(image_path)
        # convert to tensor
        image = T.functional.to_tensor(image)
        # recover real and input image
        image_width = image.shape[2]
        real_image = image[:, :, : image_width // 2]
        input_image = image[:, :, image_width // 2 :]

        if self.transform:
            # apply data transformation
            real_image = self.transform(real_image)
            input_image = self.transform(input_image)

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
    plt.figure(figsize=(10,5))

    # recover image of each batch of size 1
    display_list = [input[0], real[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot
        to_display = np.clip(display_list[i].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imshow(to_display)
        plt.axis('off')
    plt.show()

def train(model, n_epochs, display_step, save_step, dataloaders, filename, lr = 2e-4, lbd = 200):
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
    display_step            : number of epochs between two displays of images
    save_step               : number of epochs between two saves of the model
    dataloaders             : dataloader to use for training
    filename                : string, a filename to give to weights
    lr                      : learning rate
    lbd                     : l1 loss weight
    Returns
    -------------
    History of training (dict)
    """
    def compute_gen_loss(real_images, conditioned_images):
        """ Compute generator loss. """
        # compute adversarial loss
        fake_images = model.generator(conditioned_images)
        disc_logits = model.discriminator(fake_images, conditioned_images)
        adversarial_loss = BCEWithLogitsLoss()(disc_logits, torch.ones_like(disc_logits))
        # compute reconstruction loss
        recon_loss = L1Loss()(fake_images, real_images)
        return adversarial_loss + lbd * recon_loss, adversarial_loss, recon_loss

    def compute_disc_loss(real_images, conditioned_images):
        """ Compute discriminator loss. """
        fake_images = model.generator(conditioned_images).detach()
        fake_logits = model.discriminator(fake_images, conditioned_images)

        real_logits = model.discriminator(real_images, conditioned_images)

        fake_loss = BCEWithLogitsLoss()(fake_logits, torch.zeros_like(fake_logits))
        real_loss = BCEWithLogitsLoss()(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    # compute dataset length
    n = len(dataloaders['train'])

    # initalize optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    # initialize timer
    since = time.time()

    # switch to eval mode and disable grad tracking
    model.eval()
    with torch.no_grad():
        input_val, real_val = next(iter(dataloaders['val']))
        generate_images(model = model, input = input_val, real = real_val)
    # switch back to train mode
    model.train()

    # instantiate history array
    history = {'gen_loss' : [], 'gan_loss' : [], 'l1_loss' : [], 'disc_loss' : []}

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        # set epoch losses to 0
        epoch_gen_loss = 0
        epoch_gan_loss = 0
        epoch_l1_loss = 0
        epoch_disc_loss = 0

        for input, real in dataloaders['train']:
            # send tensors to device
            input = input.to(device)
            real = real.to(device)

            # generator step
            optimizer_G.zero_grad()
            gen_loss, gan_loss, l1_loss = compute_gen_loss(real, input)
            gen_loss.backward()
            optimizer_G.step()

            # discriminator step
            optimizer_D.zero_grad()
            disc_loss = compute_disc_loss(real, input)
            disc_loss.backward()
            optimizer_D.step()

            # update epoch losses
            epoch_gen_loss += gen_loss
            epoch_gan_loss += gan_loss
            epoch_l1_loss += l1_loss
            epoch_disc_loss += disc_loss
            
        # print losses
        print(f'gen_loss: {epoch_gen_loss/n:.4f}, gan_loss: {epoch_gan_loss/n:.4f}, l1_loss: {epoch_l1_loss/n:.4f}, disc_loss: {epoch_disc_loss/n:.4f}.')
        history['gen_loss'].append(epoch_gen_loss.item()/n)
        history['gan_loss'].append(epoch_gan_loss.item()/n)
        history['l1_loss'].append(epoch_l1_loss.item()/n)
        history['disc_loss'].append(epoch_disc_loss.item()/n)


        if (epoch + 1) % display_step == 0:
            # switch to eval mode and disable grad tracking
            model.eval()
            with torch.no_grad():
                input_val, real_val = next(iter(dataloaders['val']))
                generate_images(model = model, input = input_val, real = real_val)
            # switch back to train mode
            model.train()

        if (epoch + 1) % save_step ==0:
            # save model weights
            print('saving model weights ...')
            torch.save(model.state_dict(), weights_path + '/' + filename + '_ep' + str(epoch) + '.pkl')

        # line break for better readability
        print('\n')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return history


def plot_and_save_history(history, filename, title = None):
    """
    Description
    -------------
    Plots and saves history
    Parameters
    -------------
    history     : a dictionary of metrics to plot with keys 'gen_loss', 'gan_loss', 'l1_loss' and 'disc_loss'
    filename    : string, a filename 
    title       : title for the plot
    """

    # save history 
    history_file = open(history_path + '/' + filename + '.pkl', "wb")
    pickle.dump(history, history_file)
    history_file.close()

    fig, axs = plt.subplots(2, 2, figsize=(8,7))

    colors = ['blue', 'green', 'red', 'purple']

    # plot
    for i, ax in enumerate(axs.reshape(-1)):
        ax.grid()
        ax.set_xlabel('epoch')
        ylab = list(history.keys())[i]
        ax.set_ylabel(ylab)
        ax.plot(history[ylab], color = colors[i])

    # add title
    if title:
        plt.subplots_adjust(wspace=0.5)
        plt.suptitle(title)
    
    # save plot and show
    plt.savefig(history_path + '/' + filename + '.png')
    plt.show()