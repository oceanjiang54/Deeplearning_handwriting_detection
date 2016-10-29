#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 00:11:27 2016

@author: rongdilin
"""
import tensorflow as tf
import numpy as np
import random as rd

# Import data
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')

#######################tensorflow################################
# weight initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
#def max_pool_2x2(x):
#	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
	return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1], padding='SAME')
 
def next_batch(batch_size):
    """Return the next `batch_size` examples from this data set."""
    _images_a = []
    _images_b = []
    _labels = []
    sample1 = rd.sample(xrange(len(train_data)), batch_size)
    sample2 = rd.sample(xrange(len(train_data)), batch_size)
    
    for i, j in zip(sample1, sample2):
            first = train_data[i]
         	#print(person1)    
           #no1=train_data[i][0]
            label_a = train_label[i]
        	#print(label_a)
            _images_a.append(first)

            second = train_data[j]
        	#no2=train_data[i][0][1][1]
            label_b = train_label[j]
            _images_b.append(second)           
            if label_a == label_b:
               _labels.append([0,1])
            else:
               _labels.append([1,0])
#		_labels.append(label)
    return [_images_a, _images_b, _labels]

def get_testset():
    """Return the next `batch_size` examples from this data set."""
    _images_a = []
    _images_b = []
    _labels = []
#    sample1 = rd.sample(xrange(len(test_data)), 1)
#    sample2 = rd.sample(xrange(len(test_data)), 1)
    
    #sample1 = rd.sample(xrange(len(test_data)), 10)
    
    for i in xrange(len(test_data) - 1):
            first = train_data[i]
         	#print(person1)    
           #no1=train_data[i][0]
            label_a = train_label[i]
        	#print(label_a)
            _images_a.append(first)
            j = i + 1
            second = train_data[j]
        	#no2=train_data[i][0][1][1]
            label_b = train_label[j]
            _images_b.append(second)           
            if label_a == label_b:
               _labels.append([0,1])
            else:
               _labels.append([1,0])
#		_labels.append(label)
    return [_images_a, _images_b, _labels]    
               
# Create the model
# placeholder
#115000 is the dimensionality of a single flattened 230 by 500 pixel MNIST image, 
#and None indicates that the first dimension, corresponding to the batch size, can be of any size. 



#reset the memory
#tf.reset_default_graph()

x_a = tf.placeholder("float", [None, 2048])
x_b = tf.placeholder("float", [None, 2048])
y_ = tf.placeholder("float", [None, 2])

x_a_image = tf.reshape(x_a, [-1, 32, 64, 1])
x_b_image = tf.reshape(x_b, [-1, 32, 64, 1])
# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# densely connected layer
w_fc1 = weight_variable([2*8*16*32, 1024])
b_fc1 = bias_variable([1024])
# readout layer
w_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

# dropout
keep_prob = tf.placeholder("float")
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

h_conv1_a = tf.nn.relu(conv2d(x_a_image, w_conv1) + b_conv1)
h_pool1_a = max_pool_4x4(h_conv1_a)
h_pool1_flat_a = tf.reshape(h_pool1_a, [-1, 8*16*32])

h_conv1_b = tf.nn.relu(conv2d(x_b_image, w_conv1) + b_conv1)
h_pool1_b = max_pool_4x4(h_conv1_b)
h_pool1_flat_b = tf.reshape(h_pool1_b, [-1, 8*16*32])


#model logic done
h_fc1_concat=tf.concat(1, [h_pool1_flat_a, h_pool1_flat_b])
h_fc1=tf.nn.relu(tf.matmul(h_fc1_concat,w_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

y_conv=tf.matmul(h_fc1_drop, w_fc2) +b_fc2

# train and evaluate the model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
#train done

#test the model
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#test done

batch_size = 100

sess = tf.Session()

sess.run(tf.initialize_all_variables())
testset=get_testset()

outfile=open('record.txt','w')

for i in xrange(1,5001):
	print 'start',i
	batch=next_batch(batch_size)
     
	sess.run(train_step,feed_dict={x_a: np.array(batch[0]), x_b:np.array(batch[1]), y_: np.array(batch[2]), keep_prob: 0.5})
	if i%200==0:
		#sess.run(train_step,feed_dict={x_a: np.array(testset[0]), x_b:np.array(testset[1]), y_:np.array(testset[2]), keep_prob: 1.0})
		result=sess.run(accuracy,feed_dict={x_a: np.array(testset[0]), x_b:np.array(testset[1]), y_:np.array(testset[2]), keep_prob: 1.0})
		print i, result
		outfile.write('%d:%f\n'%(i,result))
outfile.close()
  