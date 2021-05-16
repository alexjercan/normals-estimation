# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt

from pathlib import Path


def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def load_normal(path):
    img = exr2normal(path)  # 3 channel normal
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def exr2normal(path):
    if not os.path.isfile(path):
        return None

    return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * 2 - 1


def plot_predictions(images, predictions, paths):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200

    normal_ps = predictions

    normal_ps = (normal_ps.cpu().numpy() + 1) / 2

    for img, normal_p, path in zip(images, normal_ps, paths):
        normal = normal_p.transpose(1, 2, 0)
        m = max(img.shape[:-1])
        normal = A.resize(normal, width=m, height=m, interpolation=cv2.INTER_NEAREST)
        normal = A.center_crop(normal, *img.shape[:-1])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(path)
        ax1.axis('off')
        ax1.imshow(img)
        ax2.axis('off')
        ax2.imshow(normal)
        plt.show()

def save_predictions(images, predictions, paths):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200

    normal_ps = predictions
    normal_ps = (normal_ps.cpu().numpy() + 1) / 2

    for img, normal_p, path in zip(images, normal_ps, paths):
        normal = normal_p.transpose(1, 2, 0)
        m = max(img.shape[:-1])
        normal = A.resize(normal, width=m, height=m, interpolation=cv2.INTER_NEAREST)
        normal = A.center_crop(normal, *img.shape[:-1])

        normal_path = str(Path(path).with_suffix(".exr"))

        cv2.imwrite(normal_path, normal)

        plt.axis('off')
        plt.imshow(normal)
        plt.savefig(str(Path(path).with_suffix(".png")))
        plt.close()
