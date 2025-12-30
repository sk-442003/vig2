"""Simple augmentation helpers for image, audio and text.
These are small, dependency-light augmentations intended for quick experiments.
"""
import random
import os
from typing import Tuple
import numpy as np

# Image augmentations use PIL and torchvision when available
try:
    from PIL import Image, ImageEnhance
    import torchvision.transforms as T
    _HAVE_TORCHVISION = True
except Exception:
    Image = None
    T = None
    _HAVE_TORCHVISION = False

# Audio augmentations use librosa when available
try:
    import librosa
    _HAVE_LIBROSA = True
except Exception:
    _HAVE_LIBROSA = False


# --- Image augmentations ---
def random_image_jitter(img):
    """Apply random color/brightness/contrast jitter using PIL or pass-through."""
    if not _HAVE_TORCHVISION or Image is None:
        return img
    # Pic is a PIL Image
    ops = [lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
           lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.9, 1.1)),
           lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.9, 1.1))]
    op = random.choice(ops)
    return op(img)


def random_image_flip(img):
    if not _HAVE_TORCHVISION or Image is None:
        return img
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# --- Audio augmentations ---
def add_noise(y: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y)) * noise_level
    return y + noise


def time_shift(y: np.ndarray, shift_max=0.2) -> np.ndarray:
    if not _HAVE_LIBROSA:
        return y
    shift = int(random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = 1.0) -> np.ndarray:
    if not _HAVE_LIBROSA:
        return y
    return librosa.effects.pitch_shift(y, sr, n_steps)


def stretch_time(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    if not _HAVE_LIBROSA:
        return y
    return librosa.effects.time_stretch(y, rate)


# --- Text augmentations ---

def random_deletion(text: str, p: float = 0.1) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    new = [w for w in words if random.random() > p]
    if len(new) == 0:
        return words[random.randrange(len(words))]
    return " ".join(new)


def random_swap(text: str, n_swaps: int = 1) -> str:
    words = text.split()
    for _ in range(n_swaps):
        i = random.randrange(len(words))
        j = random.randrange(len(words))
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


# --- Helpers to augment datasets on disk (simple) ---

def augment_audio_file(in_path: str, out_path: str, sr: int = 22050):
    """Load audio, apply random augmentation, and save to out_path (requires librosa and soundfile)."""
    if not _HAVE_LIBROSA:
        raise RuntimeError("librosa is required for audio augmentation")
    import soundfile as sf
    y, sr = librosa.load(in_path, sr=sr)
    if random.random() < 0.4:
        y = add_noise(y, noise_level=random.uniform(0.002, 0.01))
    if random.random() < 0.3:
        y = time_shift(y, shift_max=0.2)
    if random.random() < 0.2:
        y = pitch_shift(y, sr, n_steps=random.uniform(-1.5, 1.5))
    # clip
    y = np.clip(y, -1.0, 1.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, sr)


def augment_image_file(in_path: str, out_path: str):
    if Image is None:
        raise RuntimeError("PIL is required for image augmentation")
    img = Image.open(in_path).convert('RGB')
    if random.random() < 0.5:
        img = random_image_flip(img)
    if random.random() < 0.5:
        img = random_image_jitter(img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def augment_text_sample(text: str) -> str:
    # apply a random simple augmentation
    if random.random() < 0.3:
        return random_deletion(text, p=0.1)
    if random.random() < 0.3:
        return random_swap(text, n_swaps=1)
    return text
