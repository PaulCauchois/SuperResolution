import streamlit as st
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchsr.models import ninasr_b0
import torch
from dotenv import load_dotenv
from pathlib import Path

from utils.common import *
from scripts.SRCNN.model import SRCNN
import argparse


dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

def fix_image(img):
    image = Image.open(img).convert("RGB")
    image_tensor = to_tensor(image)
    model = ninasr_b0(scale=int(os.getenv('UPSCALING')), pretrained=True)
    output_tensor = model(image_tensor.unsqueeze(0))
    output_image = to_pil_image(output_tensor.squeeze(0).clamp(0, 1))
    return output_image


def fix_image_from_SRCNN(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="", help='-')
    parser.add_argument('--architecture', type=str, default="915", help='-')
    FLAGS, unparsed = parser.parse_known_args()

    image = read_image(img)

    architecture = FLAGS.architecture
    if architecture not in ["915", "935", "955"]:
        raise ValueError("architecture must be 915, 935, 955")
    scale = int(os.getenv('UPSCALING'))
    if scale not in [2, 3, 4]:
        raise ValueError("must be 2, 3 or 4")
    ckpt_path = FLAGS.ckpt_path
    if (ckpt_path == "") or (ckpt_path == "default"):
        ckpt_path = f"../../data/checkpoint/SRCNN{architecture}/SRCNN-{architecture}.pt"
    sigma = 0.3 if scale == 2 else 0.2
    pad = int(architecture[1]) // 2 + 6


    lr_image = gaussian_blur(image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = rgb2ycbcr(bicubic_image)
    bicubic_image = norm01(bicubic_image)
    bicubic_image = torch.unsqueeze(bicubic_image, dim=0)

    model = SRCNN(architecture, device)
    model.load_weights(ckpt_path)
    with torch.no_grad():
        bicubic_image = bicubic_image.to(device)
        sr_image = model.predict(bicubic_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)
    sr_image = ycbcr2rgb(sr_image)

    output_image = Image.fromarray(sr_image.transpose(1, 2, 0))

    return output_image


def app():
    st.title("Image Super Resolution")
    st.write("Upload an image to see its super resolution version")

    col1, col2 = st.columns(2)
    my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        col1.image(my_upload,caption="Original Image", use_column_width=True)

        if st.button("Model Nina SR"):
            fixed = fix_image(my_upload)
            col2.image(fixed, caption="Fixed Image - Model Nina SR", use_column_width=True)
        elif st.button("Model SRCNN"):
            fixed = fix_image_from_SRCNN(my_upload)
            col2.image(fixed, caption="Fixed Image - Model SRCNN", use_column_width=True)
        elif st.button("Model Paul "):
            fixed = fix_image_from_SRCNN(my_upload)
            col2.image(fixed, caption="Fixed Image - Model Paul", use_column_width=True)


if __name__ == "__main__":
    app()
