import pickle

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from scipy.signal import medfilt
from scipy.ndimage import median_filter, gaussian_filter
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float

from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm

from spatial_monet import EncoderNet, DecoderNet, differentiable_sampling, reconstruction_likelihood
from datasets import Atari

data_dir = '../master_thesis_code/src/data/demon_attack/static_gym_no_white'


def main():
    transform = transforms.Compose([transforms.Lambda(lambda x: transforms.functional.crop(x, 16, 0, 170, 160)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = Atari(data_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=1,
                                              shuffle=True)
    small_images = []
    for image in tqdm(list(trainloader)):
        np_image = image[0].detach().numpy()[0]
        np_image = np.transpose(np_image, (1,2,0))
        image_max = gaussian_filter(np_image, 3)
        coordinates = peak_local_max(image_max, min_distance=50, exclude_border=False)
        
        shape = (32,32)
        transform = transforms.Compose([transforms.Resize(shape),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.float())])
        for x, y, _ in coordinates:
            try:
                crop = (np_image[x-10:x+10, y-10:y+10] * 255).astype('uint8')
                crop = Image.fromarray(crop)
                crop = transform(crop)
                small_images.append(crop)
            except ValueError:
                pass
    pickle.dump(torch.stack(small_images), open('small_images', 'wb'))



if __name__ == '__main__':
    main()
