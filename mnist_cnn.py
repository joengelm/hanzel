"""
mnist_cnn.py

Implement a Convolutional Neural Network for the MNIST classification task. Consists of two
convolution + max pooling layers with ReLU activation, plus 2 fully-connected layers.
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from hanzel.net import Net

defaults = {
    'image_size': 28,      # width (also height) of square MNIST image (in pixels)
    'num_classes': 10,      # number of output classes
    'patch_size': 5,        # patch size, for convolution
    'conv1_channel': 32,    # number of features to compute for first convolution layer
    'conv2_channel': 64,    # number of features to compute for second convolution layer
    'hidden_size': 1024,    # size of the fully-connected hidden layer
    'learning_rate': 1e-4   # learning rate for the Adam Optimizer
}

class MnistCNN(Net):
    def setup(self):
        for param, default in defaults.iteritems():
            if param not in self.config:
                self.config[param] = default
        self.config.after_pool_size = self.config.image_size / (2 * 2)

        # Initialize Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.image_size * self.config.image_size])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes])
        self.dropout = tf.placeholder(tf.float32)

        # Reshape Image to be of shape [batch, width, height, channel]
        self.x_image = tf.reshape(self.x, [-1, self.config.image_size, self.config.image_size, 1])

        self.W_conv1 = self.weight_variable([self.config.patch_size, self.config.patch_size, 1, self.config.conv1_channel], "W_conv1")
        self.b_conv1 = self.weight_variable([self.config.conv1_channel], "B_conv1")

        self.W_conv2 = self.weight_variable([self.config.patch_size, self.config.patch_size, self.config.conv1_channel, self.config.conv2_channel], "W_conv2")
        self.b_conv2 = self.bias_variable([self.config.conv2_channel], "B_conv2")

        self.W_fc1 = self.weight_variable([self.config.after_pool_size * self.config.after_pool_size * self.config.conv2_channel, self.config.hidden_size], "W_fc1")
        self.b_fc1 = self.bias_variable([self.config.hidden_size], "B_fc1")

        self.W_fc2 = self.weight_variable([self.config.hidden_size, self.config.num_classes], "W_fc2")
        self.b_fc2 = self.bias_variable([self.config.num_classes], "B_fc2")

    def inference(self):
        # Convolution and Pooling 1
        h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Convolution and Pooling 2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully Connected (Hidden) Layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, self.config.after_pool_size * self.config.after_pool_size * self.config.conv2_channel])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout self.dropout fraction of the units
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)

        # Fully Connected (Logits) Layer
        logits = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
        return logits

    def loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

    def train(self, X, Y):
        self.train_op.run(feed_dict={self.x: X, self.y: Y, self.dropout: 0.5}, session=self.session)

    def test(self, x, y):
        # Create operations for computing the accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.accuracy.eval(feed_dict={self.x: x, self.y: y, self.dropout: 1.0}, session=self.session)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Main Training Block
if __name__ == "__main__":
    # Read in data, write gzip files to "data/" directory
    mnist_data = input_data.read_data_sets("data/mnist/", one_hot=True)
    batch_size = 50

    config = {}     # Use defaults (at top of file)

    # Start Tensorflow Session
    with MnistCNN(config) as net:
        # Start Training Loop
        for i in range(2000):
            batch = mnist_data.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = net.test(batch[0], batch[1])
                print "Step %d, Training Accuracy: %g" % (i, train_accuracy)
            net.train(batch[0], batch[1])

        # Evaluate Test Accuracy
        print "Final Test Accuracy: %g" % net.test(mnist_data.test.images, mnist_data.test.labels)
