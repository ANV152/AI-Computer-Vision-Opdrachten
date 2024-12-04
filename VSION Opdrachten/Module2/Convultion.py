import matplotlib.pyplot as plt
from skimage import data, filters
import scipy
import numpy as np
from scipy import ndimage
import math

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')  # Verberg de assen
    plt.show()

image = data.camera()
# Toon de originele afbeelding
show_image(image, title='Originele Afbeelding')
image = image / 255.0

#--------------------------------------- Zelf gemaakte filters ---------------------------------------------------------------------
mask_size = 3
sterkte = 2
sigma = (mask_size - 1) / (2 * math.sqrt(2 * math.log(2)))

def gaussian_matrix(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    gaussian = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return gaussian / gaussian.sum()

gaussian_mask = gaussian_matrix(mask_size, sigma)

print("Gaussian: ", gaussian_mask)

y, x = np.mgrid[-mask_size // 2 + 1:mask_size // 2 + 1, -mask_size // 2 + 1:mask_size // 2 + 1]
sobel_x = x
sobel_x[1] *= 2
sobel_y = y
sobel_y[0][1] *= 2
sobel_y[2][1] *= 2

print(sobel_x)
print(sobel_y)

la_mask_x = np.array([[0, 1, 0], [1, -4., 1], [0, 1, 0]])

# Generate a 5x5 Laplacian kernel with a circular shape
testimage = np.zeros((5, 5))
testimage[2][2] = 1.0
laplacian_kernel = filters.laplace(testimage, 3)

print(laplacian_kernel)

newimage = scipy.ndimage.convolve(image, gaussian_mask)
show_image(newimage, title='Gaussian Filtered Image')

newimage_x = scipy.ndimage.convolve(newimage, sobel_x)
newimage_y = scipy.ndimage.convolve(newimage, sobel_y)

new_image_g = (newimage_x**2 + newimage_y**2)**0.5

show_image(newimage_x, title='Sobel X Filtered Image')
show_image(newimage_y, title='Sobel Y Filtered Image')
show_image(new_image_g, title='Gradient Magnitude Image')

# Library filters
print(image.shape)
ga_filter = filters.gaussian(image, 1)
la_filter = filters.laplace(ga_filter, 3)

show_image(ga_filter, title='Gaussian Filter (Library)')
show_image(la_filter, title='Laplacian Filter (Library)')
