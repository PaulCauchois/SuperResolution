import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from torchsr.models import ninasr_b0
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from skimage.metrics import mean_squared_error
import os
from dotenv import load_dotenv
from pathlib import Path

'''
WITH UPSCALE 2 :
Average MSE: 0.06274
Average PSNR: 13.144570114108731

WITH UPSCALE 4 : 
Average MSE: 0.06369
Average PSNR: 13.084004002356544

'''


dotenv_path = Path('../../.env')
load_dotenv(dotenv_path=dotenv_path)

# LOAD THE PRE-TRAINED NINASR MODEL
model = ninasr_b0(scale=int(os.getenv('UPSCALING')), pretrained=True)

# SET THE DIRECTORIES FOR THE TEST & ORIGINAL IMAGES
test_dir = "../../data/downscale"
original_dir = "../../data/original"

mse_list = []
psnr_list = []

for i in range(1, int(os.getenv('NB_IMAGE')) + 1):
    # SET FILE PATHS FOR TEST & ORIGINAL IMAGES
    img_path = f"{test_dir}/img{i}_downscale.png"
    img_hr_path = f"{original_dir}/img{i}.png"

    # LOAD THE TEST AND ORIGINAL IMAGES
    img_test = Image.open(img_path).convert("RGB")
    img_test = to_tensor(img_test)
    img_hr = Image.open(img_hr_path).convert("RGB")
    img_hr = to_tensor(img_hr)

    # USING THE MODEL
    with torch.no_grad():
        sr_img = model(img_test.unsqueeze(0)).squeeze().clamp(0, 1)

    sr_img = sr_img.permute(1, 2, 0)
    sr_img = Image.fromarray((sr_img.cpu().numpy() * 255).astype('uint8'))
    sr_img = sr_img.resize((img_hr.shape[2], img_hr.shape[1]), resample=Image.BICUBIC)
    sr_img = to_tensor(np.array(sr_img)).permute(0, 1, 2)

    print('sr_img.shape', sr_img.shape)
    print('hr_img.shape', img_hr.shape)

    # METRICS
    mse_mesure = mean_squared_error(img_hr.cpu().numpy(), sr_img.cpu().numpy()).item()
    psnr_mesure = psnr(img_hr.cpu().numpy(), sr_img.cpu().numpy(), data_range=1.0)

    # SAVE THE UPSCALED IMAGE OUTPUT
    output_image = to_pil_image(sr_img.squeeze(0).clamp(0, 1))
    output_image.save('../../data/output/img{}_output_from_NINASR.png'.format(i))

    print('MSE : ', mse_mesure)
    print('psnr : ', psnr_mesure)

    psnr_list.append(psnr_mesure)
    mse_list.append(mse_mesure)

# CALCULATE THE AVERAGE MSE AND PSNR OVER ALL IMAGES
avg_psnr = sum(psnr_list) / len(psnr_list)
avg_mse = sum(mse_list) / len(mse_list)
print(f"Average MSE: {avg_mse:.5f}")
print(f"Average PSNR: {avg_psnr}")
