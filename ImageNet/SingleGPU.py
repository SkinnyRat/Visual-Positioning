# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A binary to train CIFAR-10 using a single GPU.
Accuracy:
CIFAR10_train.py achieves ~86% accuracy after 100K steps (256 epochs of data) as judged by CIFAR10_eval.py.

Speed: 
With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import tensorflow as tf
import CIFAR10
from   datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


parser = CIFAR10.parser

parser.add_argument('--train_dir',  type=str,  default='/HOME/MODEL', help='Directory for event logs and checkpoints.')

parser.add_argument('--max_steps',  type=int,  default=200000,  help='Number of batches to run.')

parser.add_argument('--log_device', type=bool, default=False,   help='Whether to log device placement.')

parser.add_argument('--frequency',  type=int,  default=1000,    help='How often to log results to the console.')


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    # Build a Graph that computes the logits predictions from the inference model.
    
    with tf.device('/cpu:0'):
      images, labels = CIFAR10.distorted_inputs()
    
    # Calculate loss.
    # Build a Graph that trains model with 1 batch of examples and updates model parameters.

    logits   = CIFAR10.inference(images)
    loss     = CIFAR10.loss(logits, labels)
    train_op = CIFAR10.train(loss,  global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""
      def begin(self):
        self._step = -1
        self._start_time = time.time()
      
      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)
      
      def after_run(self, run_context, run_values):
        if self._step % FLAGS.frequency == 0:
          current_time     = time.time()
          duration         = current_time - self._start_time
          self._start_time = current_time

          loss_value       = run_values.results
          examples_per_sec = FLAGS.frequency * FLAGS.batch_size / duration
          sec_per_batch    = float(duration /  FLAGS.frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
          print (format_str % (datetime.now(), self._step+1000, loss_value, examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
        hooks  = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):
  #CIFAR10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()


