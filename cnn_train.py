from __future__ import print_function

import os
import time
import tensorflow as tf

from cnn_structure import conv_net
from tfrecords_reader import read_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = './models/CNN'

# Parameters
learning_rate = 0.0005
batch_size = 256
training_iters = int(5240 * 10000 * 2 / batch_size)
save_step_num = int(training_iters / 5)
save_step = [1, save_step_num, save_step_num * 2, save_step_num * 3, save_step_num * 4, training_iters]
data_path = 'F:/go_data/training3'

# Network Parameters
dropout = 1  # Dropout, probability to keep units
global_step = tf.Variable(0, name='global_step', trainable=False)  # 计数器变量，保存模型用，设置为不需训练
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

x, y, legal_label = read_data(batch_size, data_path + '/go_training_data_*.tfrecords')
# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 17, 92], stddev=0.05)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 92, 384], stddev=0.05)),
    'wc3': tf.Variable(tf.random_normal([3, 3, 384, 512], stddev=0.05)),
    'wc4': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc5': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc6': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc7': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc8': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc9': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc10': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc11': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc12': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc13': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc14': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc15': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc16': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc17': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc18': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc19': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    'wc20': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
    'wc21': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
    # fully connected
    'wd1': tf.Variable(tf.random_normal([512, 1024], stddev=0.04)),
    # 1024 inputs, 309 outputs (class prediction)
    'wout': tf.Variable(tf.random_normal([1024, 362], stddev=1 / 1024.0))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([92])),
    'bc2': tf.Variable(tf.random_normal([384])),
    'bc3': tf.Variable(tf.random_normal([512])),
    'bc4': tf.Variable(tf.random_normal([512])),
    'bc5': tf.Variable(tf.random_normal([512])),
    'bc6': tf.Variable(tf.random_normal([512])),
    'bc7': tf.Variable(tf.random_normal([512])),
    'bc8': tf.Variable(tf.random_normal([512])),
    'bc9': tf.Variable(tf.random_normal([512])),
    'bc10': tf.Variable(tf.random_normal([512])),
    'bc11': tf.Variable(tf.random_normal([512])),
    'bc12': tf.Variable(tf.random_normal([512])),
    'bc13': tf.Variable(tf.random_normal([512])),
    'bc14': tf.Variable(tf.random_normal([512])),
    'bc15': tf.Variable(tf.random_normal([512])),
    'bc16': tf.Variable(tf.random_normal([512])),
    'bc17': tf.Variable(tf.random_normal([512])),
    'bc18': tf.Variable(tf.random_normal([512])),
    'bc19': tf.Variable(tf.random_normal([512])),
    'bc20': tf.Variable(tf.random_normal([512])),
    'bc21': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bout': tf.Variable(tf.random_normal([362]))
}

restore_var = dict(weights, **biases)

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

sc = tf.get_collection("scale")
bt = tf.get_collection("beta")
pm = tf.get_collection("pop_mean")
pv = tf.get_collection("pop_var")
for i in range(len(sc)):
    restore_var['scale' + str(i)] = sc[i]
    restore_var['beta' + str(i)] = bt[i]
    restore_var['pop_mean' + str(i)] = pm[i]
    restore_var['pop_var' + str(i)] = pv[i]

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

tf.add_to_collection('inputs', x)
tf.add_to_collection('pred', pred)

# save models
ckpt_dir = model_path + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(restore_var)

# 分配显存
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Launch the graph
with tf.Session(config=config) as sess:
    sess.run(init_op)
    start = global_step.eval()
    ckpt = tf.train.get_checkpoint_state('./play_model')
    if ckpt is not None:
        saver.restore(sess, ckpt.model_checkpoint_path)
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while step <= training_iters:
        op = sess.run(optimizer, feed_dict={keep_prob: dropout})
        if step % 1000 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={keep_prob: 1.})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(
                acc))
        if step in save_step:
            global_step.assign(step).eval()
            saver.save(sess, ckpt_dir + '/model.ckpt', global_step=global_step)
        step += 1
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
