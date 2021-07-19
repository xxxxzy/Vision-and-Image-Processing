from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.io import imread
from skimage import color
import skimage
import matplotlib.pyplot as plt
import cv2

#this imread convert png to 2 channel
img = imread('segmentation\camera.png')
image = skimage.img_as_float(img)

cv = chan_vese(image, mu=0.1, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=500,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

fig = plt.imshow(cv[0], cmap="gray")
plt.axis('off')
plt.savefig("coins_Chan-Vese %d iterations.png" % (len(cv[2])),bbox_inches = 'tight', pad_inches = 0)
plt.show()

fig = plt.imshow(cv[1], cmap="gray")
plt.axis('off')
plt.savefig("coins_Chan-Vese final level set.png",bbox_inches = 'tight', pad_inches = 0)
plt.show()

fig = plt.plot(cv[2])
plt.savefig("coins_Evolution of energy.png",bbox_inches = 'tight', pad_inches = 0)
plt.show()
