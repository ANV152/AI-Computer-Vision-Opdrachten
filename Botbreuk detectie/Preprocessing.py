import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import scipy
from skimage import data, filters, color
# from skimage import unsharp_mask
from Scripts.load_data import load_dataset, visualize_dataset

train_images, train_bbox = load_dataset('train.tfrecord')
test_images, test_bbox = load_dataset('test.tfrecord')
# Hier zijn de images al genormaliseerd van 0 naar 1 schaal
sharp_kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
def preprocess_imgs(images):
    # Eerst een lichte vervaging toepassen om ruis te verminderen
    images_blurred = filters.gaussian(images, 0.5)
    # Vervolgens een unsharp mask toepassen om de afbeelding scherper te maken
    amount = 1.0  # Dit is de sterkte van het verscherpen

    
    gray_image = color.rgb2gray(images_blurred)
    print(gray_image.shape)
    sharpened_image = scipy.ndimage.convolve(gray_image, sharp_kernel)
    # sharpened_image = cv2.filter2D(gray_image, -1, sharp_kernel)
    return sharpened_image

img_to_preprocess = preprocess_imgs(train_images[0])
plt.figure()
plt.imshow(img_to_preprocess)
plt.show()



