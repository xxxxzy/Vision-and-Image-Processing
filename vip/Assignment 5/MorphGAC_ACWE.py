import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.io import imread
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set

#store the evolution of the level sets
def store_evolution_in(lst):
    def _store(x):
        lst.append(np.copy(x))
    return _store


# Morphological ACWE
img = imread("segmentation/coins.png")
image = img_as_float(img)


# Initial level set
init_ls = checkerboard_level_set(image.shape, 6)

# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_chan_vese(image, 500, init_level_set=init_ls, smoothing=2,
                             iter_callback=callback)

fig = plt.imshow(image, cmap="gray")
plt.axis('off')
plt.contour(ls, [0.5], colors='r')
plt.savefig("Morphological ACWE segmentation.png",bbox_inches = 'tight', pad_inches = 0)
plt.show()

plt.imshow(ls, cmap="gray")
plt.axis('off')
plt.contour(evolution[2], [0.5], colors='g')
plt.contour(evolution[10], [0.5], colors='y')
plt.contour(evolution[-1], [0.5], colors='r')
plt.savefig("Morphological ACWE evolution.png",bbox_inches = 'tight', pad_inches = 0)
plt.show()





# Morphological GAC
gimage = inverse_gaussian_gradient(image)

# Initial level set
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1

# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_geodesic_active_contour(gimage, 500, init_ls, smoothing=1, balloon=-1, threshold=0.69, iter_callback=callback)


fig = plt.imshow(image, cmap="gray")
plt.axis('off')
plt.contour(ls, [0.5], colors='r')
plt.savefig("Morphological GAC segmentation.png",bbox_inches = 'tight', pad_inches = 0)
plt.show()


fig = plt.imshow(ls, cmap="gray")
plt.axis('off')
contour = plt.contour(evolution[5], [0.5], colors='g')
contour = plt.contour(evolution[100], [0.5], colors='y')
contour = plt.contour(evolution[-1], [0.5], colors='r')
plt.savefig("Morphological GAC evolution.png",bbox_inches = 'tight', pad_inches = 0)
plt.show()

