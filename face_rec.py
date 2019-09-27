# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:42:42 2019

@author: Youssef Kishk
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.simplefilter("ignore")

#************************************************************************************
def read_images():
    file_name = "att_faces"
    data_directory = ""+file_name
    
    data = np.zeros((400,10304))
    labels = np.zeros((400,1)) 
    
    i=0
    for name in os.listdir(data_directory):
        folderPath = os.path.join(data_directory,name)
        for ImageName in os.listdir(folderPath):
            Image_path = os.path.join(folderPath,ImageName)
            
            img = cv2.imread(Image_path,0)
            data[i,:] = img.flatten()
            labels[i] =  int(name)
            i+=1
    return data,labels

#************************************************************************************
    
def train_test_split(data,labels):
    train_split_value = int(data.shape[0]*(5/10))
    test_split_value = data.shape[0] - train_split_value
    
    train_data = np.zeros((train_split_value,10304))
    train_labels = np.zeros((train_split_value,1)) 
    
    test_data = np.zeros((test_split_value,10304))
    test_labels = np.zeros((test_split_value,1))
    
    #odd rows for train data, even rows for test data
    i_train=0
    i_test=0
    for i in range(data.shape[0]):
        #even
        if i%2==0:
           test_data[i_test,:] = data[i]
           test_labels[i_test] = labels[i]
           i_test+=1
        #odd
        else:
           train_data[i_train,:] = data[i]
           train_labels[i_train] = labels[i]
           i_train+=1
           
    return train_data,train_labels,test_data,test_labels

#************************************************************************************  
    
# compute the mean matrix for rach of the 40 classes
def compute_classes_mean_matrix(train_data,train_labels):
    means = np.zeros((40,10304)) 
    train_test_split_ratio = 5
    
    for i in range(1,41):
        temp = np.where(train_labels == i)[0]
        temp_sum = np.zeros((1,10304)) 
        for j in range (train_test_split_ratio):
           temp_sum += train_data[temp[j],:]        
            
        means[i-1,:] = temp_sum / train_test_split_ratio
    return means

#************************************************************************************
    
#the overall mean for all the 40 classes
#10304*1
def compute_overall_mean_matrix(classes_means):
    temp_sum = np.zeros((1,10304)) 
    for i in range(0,40):
        temp_sum +=classes_means[i,:]
    overall_mean = temp_sum / 40
    
    return overall_mean.T

#************************************************************************************
    
#the matrix of the overall scatter between all the 40 classes
def compute_between_class_scatter_matrix(classes_means,overall_mean):
    n=5
    #10304*10304
    Sb = np.zeros((classes_means.shape[1],classes_means.shape[1]))
    for i in range(classes_means.shape[0]):
        Sb = np.add(Sb,n* ((classes_means[i] - overall_mean) * (classes_means[i] - overall_mean).T))
    return Sb

#************************************************************************************
    
def compute_center_class_matrix(train_data,train_labels,classes_means):
    Z = np.zeros(train_data.shape)
    
    for i in range(train_data.shape[0]):
        Z[i,:] = train_data[i,:] - classes_means[int(train_labels[i])-1,:]

    return Z

#************************************************************************************
    
def compute_class_scatter_matrix(Z):
    S = np.zeros((10304,10304))
    S = np.dot(Z.T,Z)
    return S  

#************************************************************************************
    
def data_dimencionality_reduction(train_data,test_data):
    train_data_dimensionally_reductuted = np.zeros((200,40)) 
    test_data_dimensionally_reductuted = np.zeros((200,40)) 
    
    i=0
    for img in train_data:
        train_data_dimensionally_reductuted[i,:]=np.dot(img,eigen_vectors)
        i+=1
    i=0
    for img in test_data:
        test_data_dimensionally_reductuted[i,:] = np.dot(img,eigen_vectors)
        i+=1
        
    return train_data_dimensionally_reductuted,test_data_dimensionally_reductuted

#************************************************************************************
    
def plot_accuracy_graph(accuracy):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 25), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
    plt.ylim(50, 100)
    plt.title('Accuracy for each K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy %')
    
#************************************************************************************
#************************************************************************************ 
    
if  __name__ == '__main__':
    
    data,labels = read_images()
    print('Done images reading')
    print('-----------------------------------------------------------')
    
    train_data,train_labels,test_data,test_labels = train_test_split(data,labels)
    
    print('Done Train Test Split')
    print('-----------------------------------------------------------')
    
    classes_means = compute_classes_mean_matrix(train_data,train_labels)
    print('Done classes means computing')
    print('-----------------------------------------------------------')
    
    overall_mean = compute_overall_mean_matrix(classes_means)
    print('Done overall mean computing')
    print('-----------------------------------------------------------')
    
    S_between = compute_between_class_scatter_matrix(classes_means,overall_mean)
    print('Done between class scater matrix computing')
    print('-----------------------------------------------------------')
    
    Z = compute_center_class_matrix(train_data,train_labels,classes_means)
    print('Done center class scatter matrix computing')
    print('-----------------------------------------------------------')
    
    S_classes = compute_class_scatter_matrix(Z)
    print('Done within class scatter matrix computing')
    print('-----------------------------------------------------------')
    
    W_value = np.dot(np.linalg.inv(S_classes),S_between)
    print('Done W = S^(-1)B  computing')
    print('-----------------------------------------------------------')
    
    #40 largest eigen values
    eigen_values,eigen_vectors = scipy.linalg.eigh(W_value,eigvals=((10304-40),(10304-1)))
    print('Done eigen values and vectors computing')
    print('-----------------------------------------------------------')
    
    #reduce dimensionality of both train and test data sets
    train_data_dimensionally_reductuted,test_data_dimensionally_reductuted = data_dimencionality_reduction(train_data,test_data)
    
    
    accuracy = []
    #Apply KNN
    for i in range(1, 25):
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(train_data_dimensionally_reductuted, train_labels)
    
        test_predict = classifier.predict(test_data_dimensionally_reductuted)
        
        true_predicted_count=0
        for j in range(0,200):
            if test_predict[j] ==test_labels[j]:
                true_predicted_count+=1
        accuracy.append((true_predicted_count/200)*100)
    
    #plot graph for different K values
    plot_accuracy_graph(accuracy) 
#************************************************************************************    
#************************************************************************************
        
