"""Image augmentations and the RandomErasing operator.

Sec. IV-A: "We introduce various image augmentations to enhance the diversity of
training data, including color jitter, random horizontal flipping, random
cropping with padding, random erasing, random rotation, and random grayscale."
"""

import math
import random

import torchvision.transforms as T
from PIL import Image

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class RandomErasing(object):
    """Randomly erase a rectangular region of a tensor image.

    Reference: Zhong et al., "Random Erasing Data Augmentation".
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=MEAN):
        self.probability = probability
        self.mean = mean
        self.sl, self.sh, self.r1 = sl, sh, r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        area = img.size()[1] * img.size()[2]
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1.0 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    for c in range(3):
                        img[c, x1:x1 + h, y1:y1 + w] = self.mean[c]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


def build_transforms(img_size=(384, 128), is_train=True, aug=True):
    height, width = img_size
    if not is_train:
        return T.Compose([
            T.Resize((height, width), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    if not aug:
        return T.Compose([
            T.Resize((height, width), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    return T.Compose([
        T.Resize((height, width), interpolation=Image.BICUBIC),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        RandomErasing(probability=0.5, mean=MEAN),
    ])
