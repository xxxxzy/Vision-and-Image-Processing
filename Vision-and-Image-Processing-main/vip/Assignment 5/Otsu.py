import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



camera = '~/camera.png'
page = '~/page.png'
rocksample = '~/rocksample.png'
coins = '~/coins.png'

def Otsu(path):
    
    image = Image.open(path)
    image = image.convert('L')
    img = np.array(image)
   

    bins = np.arange(256)
    hist, _ = np.histogram(img, np.hstack((bins, np.array([256]))))
    

    N = img.size
    hist_norm = hist / N
    var = 0
    
    for th in range(255):
        
        mu0 = 0
        mu1 = 0
        w0 = np.sum(hist_norm[0:th+1])
        w1 = 1-w0
        
        for i in range(th+1):
            mu0 = mu0 + i * hist_norm[i]
            
        if w0 != 0:
            mu0 = mu0 / w0
            
        for i in range(th+1, 256):
            mu1 = mu1 + i * hist_norm[i]
            
        if w1 != 0:
            mu1 = mu1 / w1
            
        var2 = w0 * w1 * (mu0-mu1)**2
        
        if var < var2:
            var = var2
            threshold = th
            
    print('threshold:', threshold)
    
    img[img > threshold] = 255
    img[img != 255] = 0

    plt.imshow(img,'gray')
    plt.axis("off")
    plt.show()
    
    return

Otsu(camera)
Otsu(page)
Otsu(rocksample)
Otsu(coins)



def CleanNoise(path,num_iterate):
    
    noise = Otsu(path)
    row = noise.shape[0]-1
    col = noise.shape[1]-1
    
    
    
    for i in range (num_iterate):
        
        img = np.zeros(noise.shape)
        
        if noise[0][1] + noise[1][1] +noise[1][0] <= 255:
            img[0][0] = 0
        if noise[0][1] + noise[1][1] +noise[1][0] >= 510:
            img[0][0] = 255
        
        if noise[0][col-1] + noise[1][col-1] + noise[1][col] <= 255:
            img[0][col] = 0
        if noise[0][col-1] + noise[1][col-1] +noise[1][col] >= 510:
            img[0][col] = 255 
        
        if noise[row-1][0] + noise[row-1][1] +noise[row][1] <= 255:
            img[row][0] = 0
        if noise[row-1][0] + noise[row-1][1] +noise[row][1] >= 510:
            img[0][0] = 255
        
        if noise[row][col-1] + noise[row-1][col-1] +noise[row-1][col] <= 255:
            img[row][col] = 0
        if noise[row][col-1] + noise[row-1][col-1] +noise[row-1][col] >= 510:
            img[row][col] = 255 
        
              
        for j in range (1,col):
            if noise[1][j] + noise[0][j+1] + noise[0][j-1] <= 255:
                img[0][j] = 0
            if noise[1][j] + noise[0][j+1] + noise[0][j-1] >= 510:
                img[0][j] = 255
                
        for j in range (1,col):
            if noise[row-1][j] + noise[row][j+1] + noise[row][j-1] <= 255:
                img[row][j] = 0
            if noise[row-1][j] + noise[row][j+1] + noise[row][j-1] >= 510:
                img[row][j] = 255
                
                
        for i in range (1,row):
            if noise[i][1] + noise[i+1][0] + noise[i-1][0] <= 255:
                img[i][0] = 0
            if noise[i][1] + noise[i+1][0] + noise[i-1][0] >= 510:
                img[i][0] = 255
        
        for i in range (1,row):
            if noise[i][col-1] + noise[i+1][col] + noise[i-1][col] <= 255:
                img[i][col] = 0
            if noise[i][col-1] + noise[i+1][col] + noise[i-1][col] >= 510:
                img[i][col] = 255
                
        for i in range (1,row):
            for j in range (1,col-1):
                if noise[i+1][j] + noise[i-1][j] + noise[i][j+1] + noise[i][j-1] <= 255:
                    img[i][j] = 0
                if noise[i+1][j] + noise[i-1][j] + noise[i][j+1] + noise[i][j-1] >= 765:
                    img[i][j] = 255
                else:
                    img[i][j] = noise[i][j]
                    
        noise = img
    
    
        
    #print(img)
    
    plt.imshow(img,cmap = 'gray')
    plt.axis("off")
    plt.show()
    
    return

CleanNoise(coins,10)
CleanNoise(camera,10)
CleanNoise(page,10)
CleanNoise(rocksample,10)

def CleanNoiseAll(path,num_iterate):
    
    noise = Otsu(path)
    row = noise.shape[0]-1
    col = noise.shape[1]-1
    
    
    
    for i in range (num_iterate):
        
        img = np.zeros(noise.shape)
        
        if noise[0][1] + noise[1][1] +noise[1][0] == 0:
            img[0][0] = 0
        if noise[0][1] + noise[1][1] +noise[1][0] == 3*255:
            img[0][0] = 255
        else:
            img[0][0] = noise[0][0]
        
        if noise[0][col-1] + noise[1][col-1] + noise[1][col] == 0:
            img[0][col] = 0
        if noise[0][col-1] + noise[1][col-1] +noise[1][col] == 3*255:
            img[0][col] = 255 
        else:
            img[0][col] = noise[0][col]
        
        if noise[row-1][0] + noise[row-1][1] +noise[row][1] == 0:
            img[row][0] = 0
        if noise[row-1][0] + noise[row-1][1] +noise[row][1] ==3*255:
            img[row][0] = 255
        else:
            img[row][0] = noise[row][0]
            
        if noise[row][col-1] + noise[row-1][col-1] +noise[row-1][col] == 0:
            img[row][col] = 0
        if noise[row][col-1] + noise[row-1][col-1] +noise[row-1][col] == 3*255:
            img[row][col] = 255 
        else:
            img[row][col] = noise[row][col]        
              
        for j in range (1,col):
            if noise[1][j] + noise[0][j+1] + noise[0][j-1] == 0:
                img[0][j] = 0
            if noise[1][j] + noise[0][j+1] + noise[0][j-1] == 3*255:
                img[0][j] = 255
            else:
                img[0][j] = noise[0][j]
                
        for j in range (1,col):
            if noise[row-1][j] + noise[row][j+1] + noise[row][j-1] == 0:
                img[row][j] = 0
            if noise[row-1][j] + noise[row][j+1] + noise[row][j-1] == 3*255:
                img[row][j] = 255
            else:
                img[row][j] = noise[row][j]
                
                
        for i in range (1,row):
            if noise[i][1] + noise[i+1][0] + noise[i-1][0] == 255:
                img[i][0] = 0
            if noise[i][1] + noise[i+1][0] + noise[i-1][0] == 3*255:
                img[i][0] = 255
            else:
                img[i][0] = noise[i][0]
        
        for i in range (1,row):
            if noise[i][col-1] + noise[i+1][col] + noise[i-1][col] == 255:
                img[i][col] = 0
            if noise[i][col-1] + noise[i+1][col] + noise[i-1][col] == 3*255:
                img[i][col] = 255
            else:
                img[i][col] = noise[i][col]
                
        for i in range (1,row):
            for j in range (1,col-1):
                if noise[i+1][j] + noise[i-1][j] + noise[i][j+1] + noise[i][j-1] == 0:
                    img[i][j] = 0
                if noise[i+1][j] + noise[i-1][j] + noise[i][j+1] + noise[i][j-1] == 255*4:
                    img[i][j] = 255
                else:
                    img[i][j] = noise[i][j]
                    
        noise = img
    
    
        
    #print(img)
    
    plt.imshow(img,cmap = 'gray')
    plt.axis("off")
    plt.show()
    
    return

CleanNoiseAll(coins,10)
CleanNoiseAll(rocksample,50)
CleanNoiseAll(camera,10)
CleanNoiseAll(page,50)