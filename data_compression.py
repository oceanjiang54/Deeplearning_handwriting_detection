# -*- coding: utf-8 -*-
# start tensorflow interactiveSession
#import tensorflow as tf
from PIL import Image
import numpy as np
import os

#compress img in porpotion  
def resizeImg(img, dst_w, dst_h):  
   
    ori_w, ori_h = img.size  
    widthRatio = heightRatio = None  
    ratio = 1  
  
    if (ori_w and ori_w > dst_w) or (ori_h and ori_h > dst_h):  
        if dst_w and ori_w > dst_w:  
            widthRatio = float(dst_w) / ori_w  
        if dst_h and ori_h > dst_h:  
            heightRatio = float(dst_h) / ori_h  
  
        if widthRatio and heightRatio:  
            if widthRatio < heightRatio:  
                ratio = widthRatio  
            else:  
                ratio = heightRatio  
  
        if widthRatio and not heightRatio:  
            ratio = widthRatio  
  
        if heightRatio and not widthRatio:  
            ratio = heightRatio  
  
        newWidth = int(ori_w * ratio)  
        newHeight = int(ori_h * ratio)  
    else:  
        newWidth = ori_w  
        newHeight = ori_h  
    return img.resize((newWidth,newHeight),Image.ANTIALIAS)
    
'''the size of image is 50 * 100'''

train_path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/images"
train_listFileName = os.popen('find ' + train_path)
train_fileAddress = train_listFileName.readlines()

#create a matrix containing files numbers of matrix that size is [100, 100] all zero
train_data = np.array([[0 for i in range(2048)] for j in range(len(train_fileAddress)-2)])
train_label = np.ones((len(train_fileAddress)-2, 1))

pre_num = 0
writer_num = 0
k = 0
for i in xrange(0,len(train_fileAddress)-2):
    #the end of each path has a /n so remove it 
    train_imagepath = train_fileAddress[i+2][:-1]
    img = Image.open(train_imagepath)
    
    #resize the img into [50 * 100]
    #train_img = np.array(resizeImg(train_img, 50, 100))
    train_img = img.resize((32, 64), Image.ANTIALIAS)
#    print(np.shape(train_img))
#    print(np.array(train_img))
    #get the writer id (change num when path changed)
    writer_num = int(train_fileAddress[i+2][70:74]) #79：83
    
    #initial a new matrix
#    train_matrix = 255 * np.ones((50, 100))
#    train_matrix[:train_img.shape[0], :train_img.shape[1]] = train_img
    #flat matrix
    train_matrix = np.ravel(train_img)
    
    #regulariztion
    train_data[i] = np.multiply(train_matrix, 1.0 / 255.0)
    
    #assign same label for img that has same id
    if (writer_num == pre_num):
        train_label[i] = k
    else:
        k = k+1
        train_label[i] = k
        
    pre_num = writer_num
    
train_data = train_data.astype(np.float32)
#train_label = train_label.astype(np.float32)    


np.save('train_data', train_data)
np.save('train_label', train_label)


test_path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/test"
test_listFileName = os.popen('find ' + test_path)
test_fileAddress = test_listFileName.readlines()
test_data = np.array([[0 for i in range(2048)] for j in range(len(test_fileAddress) - 2)])
test_label = np.ones((len(test_fileAddress)-2, 1))
pre_num_test = 0
writer_num_test = 0
k_test = 0
for i in xrange(0,len(test_fileAddress) - 2):
    test_imagepath = test_fileAddress[i+2][:-1]
    t_img = Image.open(test_imagepath)
    
    #resize the img into [50 * 100]
    #test_img = np.array(resizeImg(test_img, 50, 100))
    test_img = t_img.resize((32, 64), Image.ANTIALIAS)

    #get the writer id (change num when path changed)
    writer_num_test = int(test_fileAddress[i+2][68:72]) #77：81
    #print writer_num
    
    #initial a new matrix
#    test_matrix = 255 * np.ones((50, 100))
#    test_matrix[:test_img.shape[0], :test_img.shape[1]] = test_img
    
    #flat matrix
    test_matrix = np.ravel(test_img)
    
    #regulariztion
    test_data[i] = np.multiply(test_matrix, 1.0 / 255.0)
    
    #assign same label for img that has same id
    if (writer_num_test == pre_num_test):
        test_label[i] = k_test
    else:
        k_test = k_test+1
        test_label[i] = k_test
        
    pre_num_test = writer_num_test
    
test_data = test_data.astype(np.float32)
#test_label = test_label.astype(np.float32) 
np.save('test_data', test_data)
np.save('test_label', test_label)
