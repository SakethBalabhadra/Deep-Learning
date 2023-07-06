import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


n_classes = 10
batch_size = 128

x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float')

keep_rate = 0.8 
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
    #                       size of window        movement of window
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random.normal([5,5,1,32])),
               'W_conv2': tf.Variable(tf.random.normal([5,5,32,64])),
               'W_fc': tf.Variable(tf.random.normal([7*7*64,1024])),
               'out': tf.Variable(tf.random.normal([1024, n_classes])),
              }
    biases = {'b_conv1': tf.Variable(tf.random.normal([32])),
              'b_conv2': tf.Variable(tf.random.normal([64])),
              'b_fc': tf.Variable(tf.random.normal([1024])),
              'out': tf.Variable(tf.random.normal([n_classes])),
             }
    x = tf.reshape(x,shape=[-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    #          rectified linear
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

    fc = tf.nn.dropout(fc,keep_rate)

    output = tf.matmul(fc,weights['out'])+biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(x_train.shape[0] / batch_size)):
                epoch_x = x_train[_ * batch_size:(_ + 1) * batch_size]
                epoch_y = y_train[_ * batch_size:(_ + 1) * batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))

train_neural_network(x)