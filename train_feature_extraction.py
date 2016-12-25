import pickle
from time import time
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('train.p', 'rb') as train_file:
    data = pickle.load(train_file)
# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'],
                                                  data['labels'],
                                                  test_size=0.1)


def one_hot_encode(labels,n_classes=43):
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

y_train, y_val = one_hot_encode(y_train), one_hot_encode(y_val)

# TODO: Define placeholders and resize operation.
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
x_resize = tf.image.resize_images(x_input, (227, 227))
y_labels = tf.placeholder(dtype=tf.float32, shape=[None, 43])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resize, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
nb_classes = 43
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=5e-2))
fc8b = tf.Variable(tf.constant(0.05, shape=[nb_classes]))
logits = tf.matmul(fc7, fc8W) + fc8b
predictions = tf.nn.softmax(logits)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.


def get_accuracy(predictions, acutals):
    prediction_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(acutals, axis=1)
    return np.mean(np.equal(prediction_labels,
                            actual_labels).astype(np.float32))

cross_entorpy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_labels)
loss = tf.reduce_mean(cross_entorpy)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# TODO: Train and evaluate the feature extraction model.

def predict_class(in_features, in_labels, sess):
    batch_size = 500
    batch_count = int(math.ceil(len(in_features)/batch_size))
    class_predictions = np.zeros_like(in_labels)
    for i in range(batch_count):
        batch_start = i*batch_size
        batch_end = batch_start+batch_size
        batch_features = in_features[batch_start : batch_end]
        batch_labels = in_labels[batch_start : batch_end]
        feed_dict = {x_input: batch_features,
                     y_labels: batch_labels
                     }
        class_predictions[batch_start:batch_end] = sess.run(predictions,
                                                            feed_dict=feed_dict)
    return class_predictions

def batch_generator(x, y, batch_size, batch_count):
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)
    x, y = x[idx], y[idx]
    for batch in range(batch_count):
        batch_start = batch*batch_size
        batch_end = batch_start + batch_size
        batch_features = x[batch_start : batch_end]
        batch_labels = y[batch_start : batch_end]
        yield batch_features, batch_labels

n_epochs = 25
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_count = int(math.ceil(len(X_train)/batch_size))
    start_time = time()
    print('Initialized')
    validation_accuracy_previous = 0
    for epoch in range(n_epochs):
        for features, labels in batch_generator(X_train, y_train,
                                                batch_size, batch_count):
            feed_dict = {x_input: features, y_labels: labels}
            _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
        if epoch % 5 == 0:
            train_predictions = predict_class(X_train, y_train, sess)
            valid_predictions = predict_class(X_val, y_val, sess)
            training_accuracy = round(
                get_accuracy(train_predictions, y_train), 3)
            validation_accuracy = round(
                get_accuracy(valid_predictions, y_val), 3)
            print("Epoch: {}, Training accuracy: {}, \
                  validation accuracy {}".format(epoch,
                                                 training_accuracy,
                                                 validation_accuracy))
    end_time = time()
    print("Time taken: {}".format(round((end_time- start_time)/60.0, 2)))
