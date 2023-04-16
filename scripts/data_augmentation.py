from skimage.util import random_noise
from skimage import io
import os
from dotenv import load_dotenv
from pathlib import Path


dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

'''
Scripts that add multiple noise to images
'''

for i in range(1, int(os.getenv('NB_IMAGE'))):
    img = io.imread(f'data/original/img{i}.png')
    noisy_image_gaussian = random_noise(img, mode='gaussian', mean=0, var=0.01)
    noisy_image_speckle = random_noise(img, mode='speckle', mean=0, var=0.01)
    noisy_image_s_and_p = random_noise(img, mode='s&p', amount=0.05)
    io.imsave(f'data/original/img{i}_noise_gaussian.png', noisy_image_gaussian)
    io.imsave(f'data/original/img{i}_noise_speckle.png', noisy_image_speckle)
    io.imsave(f'data/original/img{i}_noise_s_and_p.png', noisy_image_s_and_p)

