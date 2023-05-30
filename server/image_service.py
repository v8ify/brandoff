import torch
import cv2
import pandas as pd
import numpy as np
import uuid
import os
from typing import List, Optional, Tuple, Union
import re
import time

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import pickle

import legacy

from PIL import Image


def get_blurred_image_path(original_image: str, extension: str):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = './yolov5_v2.pt')

    results = model(original_image)

    image = cv2.imread(original_image)

    blurred_image = cv2.blur(image,(555, 555))

    for box in results.xyxy[0]: 
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])

        w = xB - xA
        h = yA - yB

        roi_corners = np.array([[(xA, yA - h), (xA, yA), (xB, yB + h), (xB, yB)]], dtype=np.int32)

        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask

        final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image, mask_inverse)

        temp_result_location = os.path.join("uploaded_images", str(uuid.uuid4()) + extension)

        cv2.imwrite(temp_result_location, final_image)

        return temp_result_location


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


#----------------------------------------------------------------------------


def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    

def convert_background(original, new_logo, xA, yA, height):
    # Define the color you want to replace the white background with
    print(original)
    new_color = original[xA - 5, yA + (height // 2)]  # replace white with red, for example
    print(new_color)

    logo_pixels = new_logo.load()

    # # Loop through each pixel in the image
    # for x in range(new_logo.size[0]):
    #     for y in range(new_logo.size[1]):
    #         # If the pixel is white, replace it with the new color
    #         if logo_pixels[x, y] == (255, 255, 255, 255):
    #             logo_pixels[x, y] = new_color + (255,)

    # Define the threshold for white pixels (here, we'll consider any pixel with an RGB value greater than 200 to be white)
    threshold = 245

    # Loop through each pixel in the image
    for x in range(new_logo.size[0]):
        for y in range(new_logo.size[1]):
            # If the pixel is white (i.e., its RGB values are all greater than the threshold), replace it with the new color
            if all(c > threshold for c in logo_pixels[x, y][:3]):
                logo_pixels[x, y] = new_color + (255,)

    pixel_list = [logo_pixels[x, y] for x in range(new_logo.size[0]) for y in range(new_logo.size[1])]

    # Create a new image with the modified pixel data
    new_img = Image.new('RGBA', new_logo.size, (255, 255, 255, 0))
    new_img.putdata(pixel_list)

    # Return the new image
    return new_img


def get_replaced_image_path(original_image: str, extension: str):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = './yolov5_v2.pt')

    results = model(original_image)

    image = cv2.imread(original_image)


    for box in results.xyxy[0]: 
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])

        w = xB - xA
        h = yB - yA

        x1, y1 = xA, yA
        width, height = w, h

        with Image.open(original_image) as original:
            seed = int(time.time()) % 1300000000
            print(seed)
            new_logo = generate_images("./network-snapshot-000960.pkl", [seed], 1, 'const', '.', (0, 0), 0, None)
            new_logo = new_logo.convert("RGBA")

            new_logo = convert_background(original.load(), new_logo, xA, yA, height)

            # resize the custom logo
            resize_custom_logo = new_logo.resize((width, height))

            # replace the portion of original logo with new custom logo
            original.paste(resize_custom_logo, (x1, y1), mask=resize_custom_logo)

            temp_result_location = os.path.join("uploaded_images", str(uuid.uuid4()) + extension)

            original.save(temp_result_location)

            return temp_result_location
    
 