import tensorflow as tf

image_size = 24
num_classes = 10
batch_size = 50
channels = 3
session = tf.Session()

with tf.name_scope('input'):
    images = tf.placeholder(tf.float32, shape=[None, image_size, image_size, channels], name='images')
    labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(0, tf.int64, name='global_step')


def conv2d(x, W, strides=1):
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')


def max_pool(x, ksize=2, strides=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME', name='pool')


def weight_variable(shape):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weights')
    print var.get_shape()
    return var


def bias_variable(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape), name='biases')
    print "bias", b.get_shape()
    return b


def flat_size(x):
    return int(x.get_shape()[1]) * int(x.get_shape()[2]) * int(x.get_shape()[3])


def inference(X, keep_prob, case):
    print ("X", X.get_shape())
    if case == "network_01":
        # conv1
        with tf.name_scope('conv1'):
            W = weight_variable([3, 3, int(X.get_shape()[-1]), 64])
            conv = conv2d(X, W)
            conv += bias_variable([64])
            conv1 = tf.nn.relu(conv)
            pool1 = max_pool(conv1)
            print ("pool1", pool1.get_shape())

        # conv2
        with tf.name_scope('conv2'):
            W = weight_variable([3, 3, int(pool1.get_shape()[-1]), 128])
            conv = conv2d(pool1, W)
            conv += bias_variable([128])
            conv2 = tf.nn.relu(conv)
            pool2 = max_pool(conv2)
            print ("pool2", pool2.get_shape())

        # conv3
        with tf.name_scope('conv3'):
            W = weight_variable([3, 3, int(pool2.get_shape()[-1]), 256])
            conv = conv2d(pool2, W)
            conv += bias_variable([256])
            conv3 = tf.nn.relu(conv)
            pool3 = max_pool(conv3)
            print ("pool3", pool3.get_shape())

        flat = tf.reshape(pool3, [-1, flat_size(pool3)])
        # fc1
        with tf.name_scope('fc1'):
            print ("flat", flat.get_shape())
            W = weight_variable([flat_size(pool3), 1024])
            b = bias_variable([1024])
            fc = tf.matmul(flat, W) + b
            fc1 = tf.nn.relu(fc)
            print ("fc1", fc1.get_shape())

        # out
        with tf.name_scope('fc2'):
            W = weight_variable([int(fc1.get_shape()[-1]), num_classes])
            b = bias_variable([num_classes])
            out = tf.matmul(fc1, W) + b
            print ("out", out.get_shape())
        return out

    elif case == "network_02":
        # conv1
        with tf.name_scope('conv1'):
            W = weight_variable([3, 3, int(X.get_shape()[-1]), 64])
            conv = conv2d(X, W)
            conv += bias_variable([64])
            conv1 = tf.nn.relu(conv)
            pool1 = max_pool(conv1)
            print ("pool1", pool1.get_shape())

        # conv2
        with tf.name_scope('conv2'):
            W = weight_variable([3, 3, int(pool1.get_shape()[-1]), 128])
            conv = conv2d(pool1, W)
            conv += bias_variable([128])
            conv2 = tf.nn.relu(conv)
            pool2 = max_pool(conv2)
            print ("pool2", pool2.get_shape())

        # conv3
        with tf.name_scope('conv3'):
            W = weight_variable([3, 3, int(pool2.get_shape()[-1]), 256])
            conv = conv2d(pool2, W)
            conv += bias_variable([256])
            conv3 = tf.nn.relu(conv)
            pool3 = max_pool(conv3)
            print ("pool3", pool3.get_shape())

        flat = tf.reshape(pool3, [-1, flat_size(pool3)])
        # fc1
        with tf.name_scope('fc1'):
            print ("flat", flat.get_shape())
            W = weight_variable([flat_size(pool3), 1024])
            b = bias_variable([1024])
            fc = tf.matmul(flat, W) + b
            fc1 = tf.nn.relu(fc)
            print ("fc1", fc1.get_shape())

        # dropout
        with tf.name_scope('dropout'):
            drop = tf.nn.dropout(fc1, keep_prob)

        # out
        with tf.name_scope('fc2'):
            W = weight_variable([int(drop.get_shape()[-1]), num_classes])
            b = bias_variable([num_classes])
            out = tf.matmul(drop, W) + b
            print ("out", out.get_shape())
        return out

    elif case == "network_03":
        # conv1
        with tf.name_scope('conv1'):
            W = weight_variable([3, 3, int(X.get_shape()[-1]), 64])
            conv = conv2d(X, W)
            conv += bias_variable([64])
            conv1 = tf.nn.relu(conv)
            pool1 = max_pool(conv1)
            print ("pool1", pool1.get_shape())

        # conv2
        with tf.name_scope('conv2'):
            W = weight_variable([3, 3, int(pool1.get_shape()[-1]), 64])
            conv = conv2d(pool1, W)
            conv += bias_variable([64])
            conv2 = tf.nn.relu(conv)
            print ("conv2", conv2.get_shape())

        # conv3
        with tf.name_scope('conv3'):
            W = weight_variable([3, 3, int(conv2.get_shape()[-1]), 128])
            conv = conv2d(conv2, W)
            conv += bias_variable([128])
            conv3 = tf.nn.relu(conv)
            pool3 = max_pool(conv3)
            print ("pool3", pool3.get_shape())

        # conv4
        with tf.name_scope('conv4'):
            W = weight_variable([3, 3, int(pool3.get_shape()[-1]), 256])
            conv = conv2d(pool3, W)
            conv += bias_variable([256])
            conv4 = tf.nn.relu(conv)
            pool4 = max_pool(conv4)
            print ("pool4", pool4.get_shape())

        flat = tf.reshape(pool4, [-1, flat_size(pool4)])
        # fc1
        with tf.name_scope('fc1'):
            print ("flat", flat.get_shape())
            W = weight_variable([flat_size(pool4), 1024])
            b = bias_variable([1024])
            fc = tf.matmul(flat, W) + b
            fc1 = tf.nn.relu(fc)
            print ("fc1", fc1.get_shape())

        # dropout
        with tf.name_scope('dropout'):
            drop = tf.nn.dropout(fc1, keep_prob)

        # out
        with tf.name_scope('fc2'):
            W = weight_variable([int(drop.get_shape()[-1]), num_classes])
            b = bias_variable([num_classes])
            out = tf.matmul(drop, W) + b
            print ("out", out.get_shape())
        return out

    elif case == "network_04":
        # conv1
        with tf.name_scope('conv1'):
            W = weight_variable([3, 3, int(X.get_shape()[-1]), 64])
            conv = conv2d(X, W)
            conv += bias_variable([64])
            conv1 = tf.nn.relu(conv)
            pool1 = max_pool(conv1)
            print ("pool1", pool1.get_shape())

        # conv2
        with tf.name_scope('conv2'):
            W = weight_variable([3, 3, int(pool1.get_shape()[-1]), 64])
            conv = conv2d(pool1, W)
            conv += bias_variable([64])
            conv2 = tf.nn.relu(conv)
            print ("conv2", conv2.get_shape())

        # conv3
        with tf.name_scope('conv3'):
            W = weight_variable([3, 3, int(conv2.get_shape()[-1]), 64])
            conv = conv2d(conv2, W)
            conv += bias_variable([64])
            conv3 = tf.nn.relu(conv)
            print ("conv3", conv3.get_shape())

        # conv4
        with tf.name_scope('conv4'):
            W = weight_variable([3, 3, int(conv3.get_shape()[-1]), 128])
            conv = conv2d(conv3, W)
            conv += bias_variable([128])
            conv4 = tf.nn.relu(conv)
            pool4 = max_pool(conv4)
            print ("pool4", pool4.get_shape())

        # conv5
        with tf.name_scope('conv5'):
            W = weight_variable([3, 3, int(pool4.get_shape()[-1]), 256])
            conv = conv2d(pool4, W)
            conv += bias_variable([256])
            conv5 = tf.nn.relu(conv)
            pool5 = max_pool(conv5)
            print ("pool5", pool5.get_shape())

        flat = tf.reshape(pool5, [-1, flat_size(pool5)])
        # fc1
        with tf.name_scope('fc1'):
            print ("flat", flat.get_shape())
            W = weight_variable([flat_size(pool5), 1024])
            b = bias_variable([1024])
            fc = tf.matmul(flat, W) + b
            fc1 = tf.nn.relu(fc)
            print ("fc1", fc1.get_shape())

        # dropout
        with tf.name_scope('dropout'):
            drop = tf.nn.dropout(fc1, keep_prob)

        # out
        with tf.name_scope('fc2'):
            W = weight_variable([int(drop.get_shape()[-1]), num_classes])
            b = bias_variable([num_classes])
            out = tf.matmul(drop, W) + b
            print ("out", out.get_shape())
        return out

    elif case == "network_05":
        # conv1
        with tf.name_scope('conv1'):
            W = weight_variable([3, 3, int(X.get_shape()[-1]), 64])
            conv = conv2d(X, W)
            conv += bias_variable([64])
            conv1 = tf.nn.relu(conv)
            pool1 = max_pool(conv1)
            print ("pool1", pool1.get_shape())

        # conv2
        with tf.name_scope('conv2'):
            W = weight_variable([3, 3, int(pool1.get_shape()[-1]), 64])
            conv = conv2d(pool1, W)
            conv += bias_variable([64])
            conv2 = tf.nn.relu(conv)
            print ("conv2", conv2.get_shape())

        # conv3
        with tf.name_scope('conv3'):
            W = weight_variable([3, 3, int(conv2.get_shape()[-1]), 128])
            conv = conv2d(conv2, W)
            conv += bias_variable([128])
            conv3 = tf.nn.relu(conv)
            pool3 = max_pool(conv3)
            print ("pool3", pool3.get_shape())

        # conv4
        with tf.name_scope('conv4'):
            W = weight_variable([3, 3, int(pool3.get_shape()[-1]), 128])
            conv = conv2d(pool3, W)
            conv += bias_variable([128])
            conv4 = tf.nn.relu(conv)
            print ("conv4", conv4.get_shape())

        # conv5
        with tf.name_scope('conv5'):
            W = weight_variable([3, 3, int(conv4.get_shape()[-1]), 256])
            conv = conv2d(conv4, W)
            conv += bias_variable([256])
            conv5 = tf.nn.relu(conv)
            pool5 = max_pool(conv5)
            print ("pool5", pool5.get_shape())

        flat = tf.reshape(pool5, [-1, flat_size(pool5)])
        # fc1
        with tf.name_scope('fc1'):
            print ("flat", flat.get_shape())
            W = weight_variable([flat_size(pool5), 1024])
            b = bias_variable([1024])
            fc = tf.matmul(flat, W) + b
            fc1 = tf.nn.relu(fc)
            print ("fc1", fc1.get_shape())

        # dropout
        with tf.name_scope('dropout'):
            drop = tf.nn.dropout(fc1, keep_prob)

        # out
        with tf.name_scope('fc2'):
            W = weight_variable([int(drop.get_shape()[-1]), num_classes])
            b = bias_variable([num_classes])
            out = tf.matmul(drop, W) + b
            print ("out", out.get_shape())
        return out


def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'))


def accuracy_score(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
    return accuracy


def train(avg_loss):
    with tf.name_scope('train'):
        return tf.train.AdamOptimizer().minimize(avg_loss, global_step)

logits = inference(images, keep_prob, "network_01")
avg_loss = loss(labels, logits)
train_op = train(avg_loss)
accuracy = accuracy_score(labels, logits)
saver = tf.train.Saver(tf.all_variables())
summary_op = tf.merge_all_summaries()
session.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter('./summary', session.graph)


def fit(X, y):
    for i in xrange(0, len(X), batch_size):
        batch_images, batch_labels = X[i:i+batch_size], y[i:i+batch_size]
        feed_dict = {
            images: batch_images,
            labels: batch_labels,
            keep_prob: 0.5
        }
        _, train_avg_loss, _global_step = session.run(fetches=[train_op, avg_loss, global_step], feed_dict=feed_dict)


def score(X, y):
    total_acc, total_loss = 0, 0
    for i in xrange(0, len(X), batch_size):
        batch_images, batch_labels = X[i:i+batch_size], y[i:i+batch_size]
        feed_dict={
            images: batch_images,
            labels: batch_labels,
            keep_prob: 1.0
        }
        acc, _avg_loss = session.run(fetches=[accuracy, avg_loss], feed_dict=feed_dict)
        total_acc += acc * len(batch_images)
        total_loss += _avg_loss * len(batch_images)
    return total_acc / len(X), total_loss / len(X)


def save(filepath):
    saver.save(session, filepath)

