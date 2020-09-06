#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:28:33 2020

@author: rohith
"""

import numpy as np
import matplotlib.pyplot as plt
    
        
if __name__ == "__main__" :
    
    rgb_values = np.ndarray(shape=(4,512*512),dtype = np.integer)
    for i in range(4):
        image = plt.imread(str(i+1)+".jpg")
        image = np.reshape(image,(512*512,3))
        print(np.shape(image))
        for j in range(512*512):
                rgb_values[i][j] = image[j][0]
    print(np.shape(rgb_values))
    #Calculate and subract mean
    T1 = [0,0,0,0]
    d1 = np.ndarray(shape=(4,512*512),dtype=np.float64)
    for i in range(4):
        T1[i] = np.sum(rgb_values[i])
        T1[i] /= (512*512)
        d1[i] = np.subtract(rgb_values[i],T1[i])
    print(np.shape(T1))
    
    #covariance matrix
    sigma = np.ndarray(shape=(4,4),dtype=np.float)
    #sigma = np.cov(d1.T,rowvar=False,bias=True)
    #print(sigma)
    
    for i in range(4):
        for j in range(4):
            sigma[i][j]=np.dot(d1[i,:],d1[j,:])/(512*512)
    print('Covariance Matrix')
    print(sigma)
    
    #sigma_inverse_r = np.linalg.inv(sigma_r)
    #sigma_inverse_nr = np.linalg.inv(sigma_nr)
    
    #eigenValues = np.linalg.eigvals(sigma)
    eigenVectors = np.linalg.eig(sigma)
    
    print("eigenValues:")
    print(eigenVectors[0])
    print("eigenVectors:")
    print(eigenVectors[1])
    
    out_image = np.dot(eigenVectors[1].T,d1)
    print(np.shape(out_image))
    out_image = np.reshape(out_image,(4,512,512))
    
    for i in range(4):
        plt.imshow(out_image[i], cmap='gray')
        plt.show()
    
    """out_image_1 = get_image()
    plt.imshow(out_image1, cmap='gray')
    plt.show()"""