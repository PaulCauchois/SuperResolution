import os
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)
path_original = "../data/original"
path_destination = "../data/downscale"
files = os.listdir(path_original)

DOWNSCALING = int(os.getenv('DOWNSCALING'))

'''
Script that downscale image 
'''

for file in files:
    if file.endswith(".jpg") or file.endswith(".png"):
        img = Image.open(os.path.join(path_original, file))
        width, height = img.size
        new_width = width // DOWNSCALING
        new_height = height // DOWNSCALING
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(os.path.join(path_destination, file))