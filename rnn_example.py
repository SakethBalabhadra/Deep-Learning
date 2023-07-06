import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.datasets import mnist
from tensorflow.python.ops import rnn, rnn_cell

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(_,_), (test_images, test_labels) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.compat.v1.placeholder('float', [None, n_chunks,chunk_size])
y = tf.compat.v1.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random.normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random.normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)


    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(x_train.shape[0] / batch_size)):
                epoch_x = x_train[_ * batch_size:(_ + 1) * batch_size]
                epoch_y = y_train[_ * batch_size:(_ + 1) * batch_size]
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss', epoch_loss)
        
        print('Shape of prediction:', prediction.shape)
        print('Shape of y:', y.shape)
        correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_images.reshape((-1,n_chunks,chunk_size)), y:test_labels}))

train_neural_network(x)