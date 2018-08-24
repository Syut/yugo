import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, s1=1, s2=1, padding='SAME', is_training=False):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, s1, s2, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    x = batch_norm(x, is_training)
    return tf.nn.relu(x)


def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    tf.add_to_collection('scale', scale)
    tf.add_to_collection('beta', beta)
    tf.add_to_collection('pop_mean', pop_mean)
    tf.add_to_collection('pop_var', pop_var)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


# Create model
def conv_net(x, weights, biases, dropout, is_training=True):
    if not is_training:
        x = tf.transpose(x, perm=[1, 2, 0])
    # Reshape input picture 19×15
    x = tf.reshape(x, shape=[-1, 19, 19, 17])

    # 1  19×19
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], is_training=is_training)
    # 2  19×19
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], is_training=is_training)
    # 3  19×19
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], is_training=is_training)
    # 4  17×19
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], padding='VALID', is_training=is_training)
    # 5  17×17
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], padding='VALID', is_training=is_training)
    # 6  15×17
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], padding='VALID', is_training=is_training)
    # 7  15×15
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'], padding='VALID', is_training=is_training)
    # 8  13×15
    conv8 = conv2d(conv7, weights['wc8'], biases['bc8'], padding='VALID', is_training=is_training)
    # 9  13×13
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'], padding='VALID', is_training=is_training)
    # 10  11×13
    conv10 = conv2d(conv9, weights['wc10'], biases['bc10'], padding='VALID', is_training=is_training)
    # 11  11×11
    conv11 = conv2d(conv10, weights['wc11'], biases['bc11'], padding='VALID', is_training=is_training)
    # 12  9×11
    conv12 = conv2d(conv11, weights['wc12'], biases['bc12'], padding='VALID', is_training=is_training)
    # 13  9×9
    conv13 = conv2d(conv12, weights['wc13'], biases['bc13'], padding='VALID', is_training=is_training)
    # 14  7×9
    conv14 = conv2d(conv13, weights['wc14'], biases['bc14'], padding='VALID', is_training=is_training)
    # 15  7×7
    conv15 = conv2d(conv14, weights['wc15'], biases['bc15'], padding='VALID', is_training=is_training)
    # 16  5×7
    conv16 = conv2d(conv15, weights['wc16'], biases['bc16'], padding='VALID', is_training=is_training)
    # 17  5×5
    conv17 = conv2d(conv16, weights['wc17'], biases['bc17'], padding='VALID', is_training=is_training)
    # 18  3×5
    conv18 = conv2d(conv17, weights['wc18'], biases['bc18'], padding='VALID', is_training=is_training)
    # 19  3×3
    conv19 = conv2d(conv18, weights['wc19'], biases['bc19'], padding='VALID', is_training=is_training)
    # 20  1×3
    conv20 = conv2d(conv19, weights['wc20'], biases['bc20'], padding='VALID', is_training=is_training)
    # 21  1×1
    conv21 = conv2d(conv20, weights['wc21'], biases['bc21'], padding='VALID', is_training=is_training)
    # Fully connected layer
    fc1 = tf.reshape(conv21, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    if is_training:
        fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    return out
