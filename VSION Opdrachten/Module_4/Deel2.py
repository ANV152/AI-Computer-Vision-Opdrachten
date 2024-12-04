from skimage import data, filters, transform
import scipy
import numpy as np
from scipy import ndimage
import math
import matplotlib.pyplot as plt
import matplotlib.image as mping

# Functie om afbeeldingen weer te geven
def show_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Laad de voorbeeldafbeelding
image = data.camera()
show_image(image, title='Originele Afbeelding')

def ImageCenter(image):
    width = image.shape[0]
    height = image.shape[1]
    center_coor = np.array([width / 2, height / 2])
    return center_coor

def makeMeHomogenous(vec):
    return np.array([vec[0], vec[1], 1])

def makeUsHomogenous(vlist):
    result = []
    for v in vlist:
        result.append(makeMeHomogenous(v))
    return np.array(result)

def rotationMatrix2D(angle):
    mtrx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)], [0, 0]])
    return makeUsHomogenous(mtrx)

def translationMatrix(v):
    return np.array([[1.0, 0.0, v[0]], [0.0, 1.0, v[1]], [0.0, 0.0, 1.0]])

stap1 = translationMatrix(-(ImageCenter(image)))
stap3 = translationMatrix(ImageCenter(image))
angl_mtrx = rotationMatrix2D(np.pi /2)
print(stap1.shape)
print(stap3.shape)
print(angl_mtrx.shape)

rot_mtrx = np.dot(stap3, np.dot(angl_mtrx, stap1))

trform = transform.AffineTransform(rot_mtrx)
tf_img = transform.warp(image, trform.inverse)
show_image(tf_img, title='Affine Transformation')

trlate_mtrx = translationMatrix([50, 1])
trform2 = transform.AffineTransform(trlate_mtrx)
tl_img = transform.warp(image, trform2.inverse)
show_image(tl_img, title='Translated Image')

def scaleMatrix(x, y):
    return np.array([[x, 0, 0], [0, y, 0], [0, 0, 1]])

scale_mtrx = scaleMatrix(1.5, 2)
trform3 = transform.AffineTransform(scale_mtrx)
scale_img = transform.warp(image, trform3.inverse)
show_image(scale_img, title='Scaled Image')
