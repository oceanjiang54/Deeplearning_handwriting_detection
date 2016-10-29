import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import data
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 50
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 50 # 1st layer num features
n_hidden_2 = 25 # 2nd layer num features
n_input = 2048 # data input (img shape: 100*100)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2
    
# batch
def next_batch(index, batch_size):
    """Return the next `batch_size` examples from this data set."""
    #index = 0
    _images = train_data
    _labels = train_label
    border = train_data.shape[0]
    start = index
    index += batch_size
    if index > border:
      # Shuffle the data
      perm = np.arange(border)
      np.random.shuffle(perm)
      _images = _images[perm]
      _labels = _labels[perm]
      # Start next epoch
      start = 0
      index = batch_size
      assert batch_size <= border
    end = index
    return _images[start:end], _labels[start:end]
          
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(train_data.shape[0] / batch_size)
    print(total_batch)
    # Training cycle
    for epoch in xrange(training_epochs):
        # Loop over all batches
        for i in xrange(total_batch):
            start = i * batch_size
            batch_xs, batch_ys = next_batch(start, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            print(c)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
     
     # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: test_data[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in xrange(examples_to_show):
        a[0][i].imshow(np.reshape(test_data[i], (32, 64)))
        a[1][i].imshow(np.reshape(encode_decode[i], (32, 64)))
    f.show()
    plt.draw()
#   plt.waitforbuttonpress()