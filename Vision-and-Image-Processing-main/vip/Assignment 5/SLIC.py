import matplotlib.pyplot as plt
from skimage import color
from skimage import morphology
from skimage import segmentation
from skimage.io import imread
import cv2

image = imread("segmentation/camera.png")
img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#print("this is rgb",image)
lum = color.rgb2gray(img)
#print("this is rbg2gray",lum)
mask = morphology.remove_small_holes(morphology.remove_small_objects(lum < 0.25,100),
    500)
mask = morphology.opening(mask, morphology.disk(3))

slic = segmentation.slic(img, n_segments=100, start_label=0)
m_slic = segmentation.slic(img, n_segments=30, mask=mask, start_label=0)


fig = plt.imshow(mask, cmap="gray")
plt.contour(mask, colors='red', linewidths=0.5)
plt.axis('off')
plt.savefig("mask",bbox_inches = 'tight', pad_inches = 0)
plt.show()

fig = plt.imshow(segmentation.mark_boundaries(img, slic))
plt.contour(mask, colors='red', linewidths=0.5)
plt.axis('off')
plt.savefig("SLIC",bbox_inches = 'tight', pad_inches = 0)
plt.show()

fig = plt.imshow(segmentation.mark_boundaries(img, m_slic))
plt.contour(mask, colors='red', linewidths=0.5)
plt.axis('off')
plt.savefig("maskSLIC",bbox_inches = 'tight', pad_inches = 0)
plt.show()
