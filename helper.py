#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from random import random
import cv2

def draw_class_distribution(class_labels):
    plt.figure(figsize=(10, 4))
    examples_per_class = np.bincount(class_labels)
    num_classes = len(examples_per_class)
    plt.bar(np.arange(num_classes), examples_per_class, 0.8, color='teal', label='Inputs per class')
    plt.xlabel('Class"')
    plt.ylabel('number of examples per class')
    plt.title("Training examples")
    plt.show()


def show_image(image):
    plt.imshow(image)
    plt.show()


def rotation(image):
    ang = random() * 360
    rows, cols, ch = image.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang, 1)
    dst = cv2.warpAffine(image, Rot_M, (cols, rows))
    return dst

def translation(image):
    rows, cols, ch = image.shape
    deltax = random() * 5
    deltay = random() * 5
    M = np.float32([[1, 0, deltax], [0, 1, deltay]])
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform(image):
    image = rotation(image)
    image = translation(image)
    image = augment_brightness_camera_images(image)
    return image

