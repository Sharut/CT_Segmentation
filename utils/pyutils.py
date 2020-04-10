import numpy as np
import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def population_mean_norm(data_dir): #Utility function to normalize input data based on mean and standard deviation of the entire dataset

    file_names = os.listdir(data_dir)
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for file in file_names:
        numpy_image = np.load(data_dir+file)
        numpy_image = np.clip(numpy_image, -1000, 1000)

        batch_mean = np.mean(numpy_image, axis=(0,1))
        batch_std0 = np.std(numpy_image, axis=(0,1))
        batch_std1 = np.std(numpy_image, axis=(0,1), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0


def show(img, title, epoch, orig): #Utility function to show figures and plots
    npimg = img.numpy()
    plt.figure()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('results/results'+str(orig)+str(epoch)+'.png')

