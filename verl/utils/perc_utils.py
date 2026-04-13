

import numpy as np
from PIL import Image, ImageFilter
from typing import Union, Tuple


def random_patch_blackening(pil_img: Image.Image, patch_size: int = 14, black_prob: float = 0.8) -> Image.Image:
    
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0
    
    return Image.fromarray(img.astype(np.uint8))


def add_gaussian_noise(pil_img: Image.Image, mean: float = 0.0, std: float = 189) -> Image.Image:
    
    img = np.array(pil_img).astype(np.float32)
    
    if img.ndim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
    
    if c == 4:
        noise = np.random.normal(mean, std, (h, w, 3))
        alpha_noise = np.zeros((h, w, 1))
        noise = np.concatenate((noise, alpha_noise), axis=-1)
    else:
        noise = np.random.normal(mean, std, img.shape)
    
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype(np.uint8)
    
    return Image.fromarray(noisy_img)


def complete_masking(pil_img: Image.Image, mask_value: Union[int, Tuple] = 128) -> Image.Image:
    
    original_array = np.array(pil_img)
    masked_array = np.full_like(original_array, fill_value=mask_value, dtype=np.uint8)
    return Image.fromarray(masked_array)


def gaussian_blur(pil_img: Image.Image, radius: Union[int, float] = 10.0) -> Image.Image:
    
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))


augment_image = random_patch_blackening