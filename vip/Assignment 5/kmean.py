import cv2
import matplotlib.pyplot as plt
import numpy as np
import random



def GetImage(file):
    
    image = cv2.imread(file)
    row, col, x = image.shape
    rgb = np.zeros((row, col, x))
    
    for i in range(row):
        for j in range(col):
            rgb[i][j] = image[i, j]
    return rgb


def Centroids(imageRGB, k):
    
    center = []
    
    for i in range(k):
        
        x, y = random.randint(0, imageRGB.shape[0]), random.randint(0, imageRGB.shape[1])
        center += [[x, y]]
        
    return center


def Distance(imageRGB, centers):
    
    region = []
    
    for i in range(imageRGB.shape[0]):
        x = []
        
        for j in range(imageRGB.shape[1]):
            temp = []
            
            for k in range(len(centers)):
                dist = np.sqrt(np.sum(np.square(imageRGB[i, j] - imageRGB[centers[k][0], centers[k][1]])))
                temp += [dist]
            x.append(np.argmin(temp))
        region.append(x)
    return region 



def EucDistance(imageRGB, CalCentercolor):
    
    region = []
    
    for i in range(imageRGB.shape[0]):
        x = []
        
        for j in range(imageRGB.shape[1]):
            temp = []
            
            for k in range(len(CalCentercolor)):
                dist = np.sqrt(np.sum(np.square(imageRGB[i, j] - CalCentercolor[k])))
                temp += [dist]
            x.append(np.argmin(temp))
        region.append(x)
    return region



def calNewCenter(features, imageRGB, k):
    
    temp = [] 
    
    for i in features:
        for j in i:
            temp.append(j)
    centercolor = [0] * k

    for i in range(len(features)):
        for j in range(len(features[i])):
            centercolor[features[i][j]] += imageRGB[i, j]
    
    for i in range(len(centercolor)):
        centercolor[i] /= temp.count(i)
        
        for j in range(len(centercolor[i])): 
            centercolor[i][j] = int(centercolor[i][j])
    return centercolor


def showImage(imageRGB, centercolor, features, k, iteration):
    
    NewImage = np.empty((len(features), len(features[0]), 3))
    
    for i in range(len(features)):
        
        for j in range(len(features[i])):
            
            NewImage[i, j] = centercolor[features[i][j]]


    plt.imshow(NewImage / 255,cmap = 'gray')
    plt.axis("off")
    plt.show()


def main():

    imageRGB = GetImage('~/page.png')

    k = 2

    InitialCenter = Centroids(imageRGB, k)
    features = Distance(imageRGB, InitialCenter)
    a = calNewCenter(features, imageRGB, k)
    iteration = 20
    
    for i in range(iteration, 0, -1):
             
        CalCentercolor = calNewCenter(features, imageRGB, k) 
        features = EucDistance(imageRGB, CalCentercolor)


        

    showImage(imageRGB, CalCentercolor, features, k, iteration)


if __name__ == '__main__':
    main()