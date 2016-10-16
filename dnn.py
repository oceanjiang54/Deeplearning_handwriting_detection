# -*- coding: utf-8 -*-
# start tensorflow interactiveSession
import tensorflow as tf
from PIL import Image
import numpy as np
import os

'''the size of image is 230 * 500'''

train_path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/images"
train_listFileName = os.popen('find ' + train_path)
train_fileAddress = train_listFileName.readlines()

#create a matrix containing files numbers of matrix that size is [230, 500] all zero
train_data = np.array([[0 for i in range(115000)] for j in range(len(train_fileAddress)-2)])
train_label = np.ones((len(train_fileAddress)-2, 1))

pre_num = 0
writer_num = 0
k = 0
for i in range(0,len(train_fileAddress)-2):
    #the end of each path has a /n so remove it 
    train_imagepath = train_fileAddress[i+2][:-1]
    train_img = np.array(Image.open(train_imagepath), dtype=float)
    
    #get the writer id (change num when path changed)
    writer_num = int(train_fileAddress[i+2][70:74]) #79：83
    
    #initial a new matrix
    train_matrix = 255 * np.ones((230, 500))
    train_matrix[:train_img.shape[0], :train_img.shape[1]] = train_img
    #flat matrix
    train_matrix = np.ravel(train_matrix)
    
    #regulariztion
    train_data[i] = np.multiply(train_matrix, 1.0 / 255.0)
    
    #assign same label for img that has same id
    if (writer_num == pre_num):
        train_label[i] = k + 1
    else:
        #k = k+1
        train_label[i] = k
        
    pre_num = writer_num
    
train_data = train_data.astype(np.float32)
train_label = train_label.astype(np.float32)    
#put two img into one, the size is 460 * 1000
#final_data = [[0 for i in range(460000)] for j in range(len(train_data)-1)]
#final_label = range(len(train_data - 1))
#
#for j in range(1, len(train_data)):
#    final_data[j] = np.append(train_data[j - 1], train_data[j])
#    print(final_data[j])
#    if train_label[j - 1] == train_label[j]:
#        final_label[j] = k + 1
#    else:
#        final_label[j] = k
np.save('train_data', train_data)
np.save('train_label', train_label)


test_path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/test"
test_listFileName = os.popen('find ' + test_path)
test_fileAddress = test_listFileName.readlines()
test_data = np.array([[0 for i in range(115000)] for j in range(len(test_fileAddress) - 2)])
test_label = np.ones((len(test_fileAddress)-2, 1))
pre_num_test = 0
writer_num_test = 0
k_test = 0
for i in range(0,len(test_fileAddress) - 2):
    test_imagepath = test_fileAddress[i+2][:-1]
    test_img = np.array(Image.open(test_imagepath), dtype=float)
    
    #get the writer id (change num when path changed)
    writer_num_test = int(test_fileAddress[i+2][68:72]) #77：81
    #print writer_num
    
    #initial a new matrix
    test_matrix = 255 * np.ones((230, 500))
    test_matrix[:test_img.shape[0], :test_img.shape[1]] = test_img
    
    #flat matrix
    test_matrix = np.ravel(test_matrix)
    
    #regulariztion
    test_data[i] = np.multiply(test_matrix, 1.0 / 255.0)
    
    #assign same label for img that has same id
    if (writer_num_test == pre_num_test):
        test_label[i] = k_test
    else:
        #k_test = k_test+1
        test_label[i] = k_test
        
    pre_num_test = writer_num_test
    
test_data = test_data.astype(np.float32)
test_label = test_label.astype(np.float32) 
np.save('test_data', test_data)
np.save('test_label', test_label)


#######################tensorflow###########


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
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def next_batch(index, batch_size):
    """Return the next `batch_size` examples from this data set."""
    #index = 0
    _images = train_data
    _labels = train_label
    border = train_data.shape[0]
    start = index
    index += batch_size
    #if index > border:
    #  # Shuffle the data
    #  perm = np.arange(border)
    #  np.random.shuffle(perm)
    #  _images = _images[perm]
    #  _labels = _labels[perm]
    #  # Start next epoch
    #  start = 0
    #  index = batch_size
    #  assert batch_size <= border
    end = index
    return _images[start:end], _labels[start:end]
    
def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(train_data))
        np.random.shuffle(idxs)
        shuf_features = train_data[idxs]
        shuf_labels = train_label[idxs]
        batch_size = 128
        for batch_idx in range(0, len(train_data), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size] #/ 255.
            images_batch = images_batch.astype(np.float32)
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            labels_batch = labels_batch.astype(np.float32)
            yield images_batch, labels_batch    
                
# Create the model
# placeholder
#115000 is the dimensionality of a single flattened 230 by 500 pixel MNIST image, 
#and None indicates that the first dimension, corresponding to the batch size, can be of any size. 
#reset the memory
tf.reset_default_graph()
sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 115000])
#print(x.dtype)
#The target output classes y_ will also consist of a 2d tensor, where each row is a one-hot 
#2-dimensional vector indicating which digit class (0 to 1) the corresponding whether character
#belongs to the same writer.
y_ = tf.placeholder("float", [None, 1])

# variables
W = tf.Variable(tf.zeros([115000,1]))
b = tf.Variable(tf.zeros([1]))


y = tf.nn.softmax(tf.matmul(x,W) + b)

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 230, 500, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([29*250*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 29*250*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())

for i in range(1000):
	    batch_xs, batch_ys = next_batch(i, 100)
	    batch_xs = batch_xs.astype(np.float32)
	    batch_ys = batch_ys.astype(np.float32)
	#iter_ = data_iterator()
 #       while True:
 #           # get a batch of data
 #           batch_xs, batch_ys = iter_.next()
 #           print (np.shape(batch_xs))
 #           print(np.shape(batch_ys))
 #           
            
	    if i%100 == 0:
              train_accuracy = accuracy.eval(feed_dict={
        x:batch_xs, y_: batch_ys, keep_prob: 1.0})
              print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

	    #feed_dict={x: batch_xs, y_: batch_ys}
	    #print(feed_dict)

	    #sess.run(train_step, feed_dict)
	    #train_step.run(feed_dict)
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                
            print "step %d, train accuracy %g" %(i, train_accuracy)
            train_step.run(feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.5})
print "test accuracy %g" % accuracy.eval(feed_dict={x:test_data, y_:test_label, keep_prob:1.0})