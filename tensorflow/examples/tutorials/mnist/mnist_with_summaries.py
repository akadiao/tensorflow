# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  
  
"""

Accuracy at step 0: 0.0794
Accuracy at step 10: 0.7096
Accuracy at step 20: 0.8259
Accuracy at step 30: 0.8651
Accuracy at step 40: 0.8848
Accuracy at step 50: 0.8902
Accuracy at step 60: 0.9016
Accuracy at step 70: 0.9065
Accuracy at step 80: 0.9081
Accuracy at step 90: 0.9099
Adding run metadata for 99
Accuracy at step 100: 0.9155
Accuracy at step 110: 0.9159
Accuracy at step 120: 0.9203
Accuracy at step 130: 0.9239
Accuracy at step 140: 0.9268
Accuracy at step 150: 0.9272
Accuracy at step 160: 0.9245
Accuracy at step 170: 0.9308
Accuracy at step 180: 0.9336
Accuracy at step 190: 0.9308
Adding run metadata for 199
Accuracy at step 200: 0.933
Accuracy at step 210: 0.9348
Accuracy at step 220: 0.9352
Accuracy at step 230: 0.9363
Accuracy at step 240: 0.937
Accuracy at step 250: 0.9399
Accuracy at step 260: 0.9403
Accuracy at step 270: 0.94
Accuracy at step 280: 0.9371
Accuracy at step 290: 0.9402
Adding run metadata for 299
Accuracy at step 300: 0.9438
Accuracy at step 310: 0.9437
Accuracy at step 320: 0.9393
Accuracy at step 330: 0.9415
Accuracy at step 340: 0.9407
Accuracy at step 350: 0.9472
Accuracy at step 360: 0.9495
Accuracy at step 370: 0.9494
Accuracy at step 380: 0.9467
Accuracy at step 390: 0.9495
Adding run metadata for 399
Accuracy at step 400: 0.9491
Accuracy at step 410: 0.9533
Accuracy at step 420: 0.9507
Accuracy at step 430: 0.9508
Accuracy at step 440: 0.9544
Accuracy at step 450: 0.9488
Accuracy at step 460: 0.9507
Accuracy at step 470: 0.9569
Accuracy at step 480: 0.958
Accuracy at step 490: 0.9584
Adding run metadata for 499
Accuracy at step 500: 0.9585
Accuracy at step 510: 0.9554
Accuracy at step 520: 0.9575
Accuracy at step 530: 0.9593
Accuracy at step 540: 0.9557
Accuracy at step 550: 0.9616
Accuracy at step 560: 0.9597
Accuracy at step 570: 0.9588
Accuracy at step 580: 0.9598
Accuracy at step 590: 0.9619
Adding run metadata for 599
Accuracy at step 600: 0.9584
Accuracy at step 610: 0.9613
Accuracy at step 620: 0.9633
Accuracy at step 630: 0.9587
Accuracy at step 640: 0.9613
Accuracy at step 650: 0.9629
Accuracy at step 660: 0.9625
Accuracy at step 670: 0.9616
Accuracy at step 680: 0.9615
Accuracy at step 690: 0.9567
Adding run metadata for 699
Accuracy at step 700: 0.9611
Accuracy at step 710: 0.962
Accuracy at step 720: 0.9634
Accuracy at step 730: 0.9611
Accuracy at step 740: 0.9608
Accuracy at step 750: 0.9626
Accuracy at step 760: 0.9621
Accuracy at step 770: 0.9646
Accuracy at step 780: 0.9647
Accuracy at step 790: 0.963
Adding run metadata for 799
Accuracy at step 800: 0.9653
Accuracy at step 810: 0.9656
Accuracy at step 820: 0.9663
Accuracy at step 830: 0.9679
Accuracy at step 840: 0.9637
Accuracy at step 850: 0.9615
Accuracy at step 860: 0.9629
Accuracy at step 870: 0.9646
Accuracy at step 880: 0.9664
Accuracy at step 890: 0.9673
Adding run metadata for 899
Accuracy at step 900: 0.9678
Accuracy at step 910: 0.9639
Accuracy at step 920: 0.9664
Accuracy at step 930: 0.9666
Accuracy at step 940: 0.9648
Accuracy at step 950: 0.9678
Accuracy at step 960: 0.9675
Accuracy at step 970: 0.9682
Accuracy at step 980: 0.9646
Accuracy at step 990: 0.9666
Adding run metadata for 999



"""
