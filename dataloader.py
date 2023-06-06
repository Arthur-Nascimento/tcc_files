import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def load_data(img_path, mask_path, gs = True, classes = 2):
    images = glob.glob(img_path,

