import math

def check_img_size(img_size, s=32):
    new_size = math.ceil(img_size / s) * s
    if new_size != img_size:
        print(f'image size {img_size} updating to {new_size}')
    return new_size

