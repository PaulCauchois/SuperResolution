from scaling_methods import *
from PIL import Image
import numpy as np
import os
import random

cwd = os.getcwd() + "\\"
in_folder = ""
id_folder = "Original"
out_folder = "Downscaled"
split = False
split_threshold = 0.8
scaling_factor = 1 / 4
nshape = None  # Leave at None to use scaling factor

if not split:
    try:
        os.mkdir(cwd + out_folder)
    except FileExistsError:
        pass
    else:
        print(f"Output folder {out_folder} doesn't exist, creating...")
else:
    try:
        os.mkdir(cwd + out_folder + "_train")
    except FileExistsError:
        pass
    else:
        print(f"Output folder {out_folder}_train doesn't exist, creating...")

    try:
        os.mkdir(cwd + out_folder + "_test")
    except FileExistsError:
        pass
    else:
        print(f"Output folder {out_folder}_test doesn't exist, creating...")

    try:
        os.mkdir(cwd + id_folder + "_train")
    except FileExistsError:
        pass
    else:
        print(f"Output folder {id_folder}_train doesn't exist, creating...")

    try:
        os.mkdir(cwd + id_folder + "_test")
    except FileExistsError:
        pass
    else:
        print(f"Output folder {id_folder}_test doesn't exist, creating...")

for file in os.listdir(cwd+in_folder):
    folder = out_folder
    print(f"Working on {file}...")
    im = Image.open(f"{cwd}{in_folder}\\{file}").convert(mode='RGB')
    arr = np.array(im)
    if nshape is None:
        nshape = (int(scaling_factor * arr.shape[0]), int(scaling_factor * arr.shape[1]))
    new_arr = bilinear(arr, nshape, step=lambda x: x ** 2 * (3 - 2 * x))  # Smoothstep
    new_im = Image.fromarray(new_arr)
    if split:
        if random.random() < 0.8:
            new_im.save(f"{out_folder}_train\\{file[:-4]}.png")
            im.save(f"{id_folder}_train\\{file[:-4]}.png")
            folder = f"{out_folder}_train"
        else:
            new_im.save(f"{out_folder}_test\\{file[:-4]}.png")
            im.save(f"{id_folder}_test\\{file[:-4]}.png")
            folder = f"{out_folder}_test"
    else:
        new_im.save(f"{cwd}{out_folder}\\{file[:-4]}.png")
    print(f"Done !\nSaved file as {file[:-4]}_{nshape[0]}_{nshape[1]}.png in folder {folder}\n----------")
