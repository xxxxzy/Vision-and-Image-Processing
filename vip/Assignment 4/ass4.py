import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing





trainpath = '~/trainset/'
testpath = '~/testset/'
Label = {'airplanes':0, 'bonsai':1, 'butterfly':2, 'car_side':3, 'chandelier':4, 'kangaroo':5, 'ketch':6, 'starfish':7, 'sunflower':8, 'watch':9}


def read_data(Label,path):
    X = []
    Y = []
    Z = []
    
    for label in os.listdir(path):
        for img_file in os.listdir(os.path.join(path, label)):
            img = cv2.imread(os.path.join(path, label, img_file)) 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray)
            Y.append(Label[label])
            Z.append(img_file)
    return X, Y, Z


X, Y ,Z= read_data(Label,trainpath)


def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)
    return image_descriptors


image_descriptors = extract_sift_features(X)


all_descriptors = []

for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)


def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

num_clusters = 1000

if not os.path.isfile('trainset/bow_dictionary150.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('trainset/bow_dictionary150.pkl', 'wb'))
else:
    BoW = pickle.load(open('trainset/bow_dictionary150.pkl', 'rb'))


def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features


X_features = create_features_bow(image_descriptors, BoW, num_clusters)
#print(len(X_features))
all_features = sum(X_features)
butterfly_features = sum(X_features[0:30])
chandelier_features = sum(X_features[30:60])
airplanes_features = sum(X_features[60:90])
starfish_features = sum(X_features[90:120])
ketch_features = sum(X_features[120:150])
bonsai_features = sum(X_features[150:180])
car_side_features = sum(X_features[180:210])
kangaroo_features = sum(X_features[210:240])
watch_features = sum(X_features[240:270])
sunflower_features = sum(X_features[270:300])


#################################################################################
######################### Train Data Histogram ##################################
#################################################################################

def plot_hist(name, feature):
    
    plt.bar(np.arange(num_clusters), feature)
    plt.xlabel('Cluster_Number')
    plt.ylabel('Frequency')
    plt.savefig(str(name))
    plt.show()
    
    return

plot_hist('allfeatures',all_features)
plot_hist('butterfly',butterfly_features)
plot_hist('chandelier',chandelier_features)
plot_hist('airplanes',airplanes_features)
plot_hist('starfish',starfish_features)
plot_hist('ketch',ketch_features)
plot_hist('bonsai',bonsai_features)
plot_hist('car_side',car_side_features)
plot_hist('kangaroo',kangaroo_features)
plot_hist('watch',watch_features)
plot_hist('sunflower',sunflower_features)


#Figure of train set
for i in range (len(X_features)):

    x= np.arange(num_clusters)
    plt.bar(x, X_features[i]) #0~39
    plt.xlabel('Cluster_Number')
    plt.ylabel('Frequency')
    #plt.xticks(x+0.7, x)
    plt.savefig("~"+str(Y[i])+str(Z[i]))
    plt.show()
  
    
#################################################################################
######################### Test Data Histogram ###################################
#################################################################################

X_T, Y_T ,Z_T= read_data(Label,testpath)

print(len(X_T))


for i in range (len(X_T)):

    img_test = X_T[i]
    img = [img_test]
    img_sift_feature = extract_sift_features(img)
    img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)

    x= np.arange(num_clusters)
    plt.bar(x, img_bow_feature[0]) 
    plt.xlabel('Cluster_Number')
    plt.ylabel('Frequency')
    plt.savefig("~/Test/"+str(Y_T[i])+str(Z_T[i]))
    #plt.xticks(x+0.7, x)
    plt.show()  


#################################################################################
################################ TF-IDF #########################################
#################################################################################

tf = np.zeros((300,num_clusters), "float32")

for i in range (300):
    sum_w = sum(X_features[i][:])
    for j in range (num_clusters):
        tf[i][j]=X_features[i][j]/sum_w


nonzero = np.zeros(num_clusters)

for i in range(num_clusters):
    num_nonzero = 0
    for j in range(300):
        if X_features[j][i] == 0:
            num_nonzero = num_nonzero+1
    nonzero[i] = num_nonzero

idf = np.log(300/(nonzero+1))

tf_idf = tf*idf


#################################################################################
######################## Retrieving of images ###################################
#################################################################################
X_train = [] 
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(tf_idf, Y, test_size=0.2, random_state=42)
#X_norm_train = preprocessing.normalize(X_train)

#def bhattacharyya(x,y):
#    return sum(np.sqrt(x*y))

#clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree',metric=bhattacharyya)
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,Y_train)
distances, indices = clf.kneighbors(X_features)

predict = clf.predict(X_features)

dis = np.zeros(300)
ind = np.zeros(300)
diss = []
for i in range(len(Y)):
    
    dis[i] = distances[i]
    if predict[i]==6:
        diss.append(dis[i])
        print(dis[i])
        print(i)
    if Y[i] == predict[i] == 6:
        print(i)


#################################################################################
######################## Classify test images ###################################
#################################################################################
X_train = [] 
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(tf_idf, Y, test_size=0.2, random_state=42)
#X_train = X_features
#Y_train = Y
X_norm_train = preprocessing.normalize(X_train)

def bhattacharyya(x,y):
    return sum(np.sqrt(x*y))

clf = neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree',metric=bhattacharyya)
#clf = neighbors.KNeighborsClassifier()
clf.fit(X_norm_train,Y_train)

cor_num = np.zeros(10)
ind = np.zeros(300)
for i in range (len(X_T)):
    
    img_test = X_T[i]
    img = [img_test]
    img_sift_feature = extract_sift_features(img)
    img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
    img_norm_bow_feature = preprocessing.normalize(img_bow_feature)
    img_predict = clf.predict(img_norm_bow_feature)

    for j in range (0,10):
        if Y_T[i] == img_predict and Y_T[i]== j:
            cor_num[j] = cor_num[j]+1

print(cor_num)
