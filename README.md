# Vision-and-Image-Processing
Course:Vision and Image Processing from KU
## Assignment 2
The program implements Gaussian filtering, Gradient magnitude and Laplacian-Gaussian filtering with σ = 1,2,4,8 and Canny edge detection with image lenna.jpg.
Filtering is done by convolutions. Among the most applied filters are the Gaussian and its first and second order derivatives. Convolution with a Gaussian itself will blurr the image. Convolutions with the two first order Gaussian derivatives provides an estimate of the gradient field from which the gradient magnitude may be derived and visualised. Convolution with the sum of the second order (unmixed) partial derivatives of a Gaussian may be used both to detect blobs and edges. Edges and blobs in images code much of the semantic information available in the images and are often used as building elements in further analysis.
### Image : lenna.jpg
## Assignment 3
The program implements basic photometric stereo and a bit less basic.
### Data
1. Beethoven.mat, 3 images, (semi-)synthetic (comes from a real model, but processed). 2. mat vase.mat, 3 images, synthetic.
3. shiny vase.mat, 3 images, synthetic.
4. shiny vase2.mat, 22 images, synthetic.
5. Buddha.mat, 10 images, real. 6. face.mat, 27 images, real.
## Assignment 4:Content Based Image Retrieval
The program implements a prototypical CBIR system.
First it generate a code book, select a set of training images. Then Extract SIFT features from the training images (ignore position, orientation and scale). The SIFT features concatenated into a matrix, one descriptor per row. Then run the k-means clustering algorithm on the subset of training descriptors to extract good prototype (visual word) clusters.
Then it implement retrieving of images by tf-ifd similarity and Bhattacharyya distance or Kullback-Leibler divergence.
### Data
The program uses the data from CalTech 101 image database. In our data set, a total of 370 images are selected from 10 categories, among which 300 are classified as the training set, and the rest 70 are classified as the test set. The 10 categories are airplanes, bonsai, butterfly, carside, chandelier, kangaroo, ketch, starfish, sunflower, watch respectively.
## Assignment 5:Segmentation
The program implements two basic segmentation algorithms:k-means algorithm and Otsu’s thresholding algorithm. It also trys some of the algorithms available in scikit-learn.
### Data
image:Cameraman,rock sample slice,Coins and page
  

