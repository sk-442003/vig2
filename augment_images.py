"""Simple image augmentation utilities to expand datasets when shortages occur.
Usage:
  python scripts/augment_images.py --in data/face/sad --out data/face/sad_aug --n 200
This will create up to `n` augmented images in the out dir (re-using input images).
"""
import os
import argparse
import random
from PIL import Image, ImageEnhance

AUG_FUNCS = []

def aug_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def aug_rotate(img, deg=10):
    return img.rotate(deg)

def aug_brightness(img, factor=1.1):
    return ImageEnhance.Brightness(img).enhance(factor)

AUG_FUNCS = [aug_flip, lambda im: aug_rotate(im, 10), lambda im: aug_rotate(im, -10), lambda im: aug_brightness(im, 0.9), lambda im: aug_brightness(im, 1.1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='indir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    imgs = [os.path.join(args.indir, f) for f in os.listdir(args.indir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not imgs:
        print('No images found in', args.indir)
        return
    cnt = 0
    i = 0
    while cnt < args.n:
        src = imgs[i % len(imgs)]
        img = Image.open(src).convert('RGB')
        # choose random augmentation chain
        func = random.choice(AUG_FUNCS)
        out = func(img)
        out_fname = os.path.join(args.out, f'aug_{cnt:05d}.png')
        out.save(out_fname)
        cnt += 1
        i += 1
    print('Wrote', cnt, 'augmented images to', args.out)

if __name__ == '__main__':
    main()
