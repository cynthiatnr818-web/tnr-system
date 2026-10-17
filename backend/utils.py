import torch
from PIL import Image
from torchvision import transforms
import base64
import io

def preprocess_image(image_path):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, bytes):
        image = Image.open(io.BytesIO(image_path)).convert('RGB')
    else:
        raise ValueError("Unsupported image format")
    
    return image

def decode_image(image_str):
    image_data = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image
