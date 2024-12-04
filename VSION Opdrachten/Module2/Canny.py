import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage import data


# Generate noisy image of a square
# image = np.zeros((128, 128), dtype=float)
# image[32:-32, 32:-32] = 1
image = data.camera()

# image = image/255.0

# image = ndi.rotate(image, 15, mode='constant')
image = ndi.gaussian_filter(image, 1)
image = random_noise(image, mode='speckle', mean=0.1)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image,low_threshold=0.01, high_threshold= 0.1 )
edges2 = feature.canny(image, sigma=2, low_threshold= 0.1, high_threshold= 0.13)
edges3 = feature.canny(image,sigma = 2, low_threshold= 0.2, high_threshold= 0.3)

# display results
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 3))

ax[0][0].imshow(image, cmap='gray')
ax[0][0].set_title('noisy image', fontsize=20)

#Hoog frequentie edges
ax[0][1].imshow(edges1, cmap='gray')
ax[0][1].set_title(r'Canny filter, $\sigma=1$ H: 0.01, L:0.1', fontsize=20)


ax[0][2].imshow(edges2, cmap='gray')
ax[0][2].set_title(r'Canny filter, $\sigma=2$ H: 0.2, L:0.1', fontsize=20)

# Laag frequentie edges
ax[1][0].imshow(edges3, cmap='gray')
ax[1][0].set_title(r'Canny filter, $\sigma=1.5$, H: 0.16, L:0.2 ', fontsize=20)

for a in range(0,len(ax)):
    for i in range(0, len(ax[a])):
        ax[a][i].axis('off')

fig.tight_layout()
plt.show()