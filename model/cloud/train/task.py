# Example of how to run.
# gcloud ml-engine jobs submit training my_job_name \
# --module-name train.task --package-path train \
# --job-dir 'gs://cxr-to-chest-ct2/gcp-training/staging' \
# --python-version 3.5 --runtime-version 1.13
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from . import model

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from google.cloud import storage

import datetime

# from IPython import embed

session_nickname = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")

def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--project',
      default='x-ray-reconstruction',
      type=str,
      help='project name')
  parser.add_argument(
      '--job-dir',
      default=os.path.join('gs://cxr-to-chest-ct2/gcp-training/', session_nickname),
      type=str,
      help='local or GCS location for writing checkpoints and exporting models')
  parser.add_argument(
      '--train-dir',
      default='tfrecords/silly-cat-2019-04-20--23h57m53s/',
      type=str,
      help='location of the training data')
  parser.add_argument(
      '--train-record-name',
      default='train',
      type=str,
      help='train tfrecord name')
  parser.add_argument(
      '--train-bucket',
      default='cxr-to-chest-ct2',
      type=str,
      help='bucket of the training data')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=20,
      help='number of times to go through the data, default=20')
  parser.add_argument(
      '--batch-size',
      default=1,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning-rate',
      default=0.01,
      type=float,
      help='learning rate for gradient descent, default=.01')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')
  return parser.parse_args()


def _parse_function(proto):
  # define your tfrecord again. Remember that you saved your image as a string.
  keys_to_features = {'image': tf.FixedLenFeature([256, 256, 1], tf.int64),
                      'volume': tf.FixedLenFeature([128 ** 3], tf.int64)}

  # Load one example
  parsed_features = tf.parse_single_example(proto, keys_to_features)

  # image = parsed_features['image']
  # image = tf.reshape(parsed_features['image'], [-1, 256, 256, 1])

  # volume = tf.reshape(parsed_features['volume'], [1, 128 ** 3])

  # return image, volume
  return parsed_features['image'], parsed_features['volume']

def create_dataset(filenames_train, perform_shuffle=False, shuffle_buffer=256, batch_size=1):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filenames_train)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the number of datapoints you want to load and shuffle
    if perform_shuffle:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    # Create an iterator
    # iterator = dataset.make_one_shot_iterator()
    # Create your tf representation of the iterator
    # image, volume = iterator.get_next()
    # Bring your picture back in shape
    # image = tf.reshape(image, [-1, 256, 256, 1])
    # Create a one hot array for your labels
    # volume = tf.reshape(volume, [1, 128 ** 3])
    return dataset
    # return image, volume

def train_and_evaluate(hparams):

  # Instantiates a client
  storage_client = storage.Client(hparams.project)
  # The name for the bucket
  bucket_name = hparams.train_bucket
  # Creates the new bucket
  bucket = storage_client.get_bucket(bucket_name)

  train_filenames = [os.path.join('gs://{}/'.format(bucket_name), f.name) for f in bucket.list_blobs(prefix=hparams.train_dir) if f.name.split('/')[-1][:len(hparams.train_record_name)] == hparams.train_record_name]

  training_dataset = create_dataset(train_filenames, batch_size=hparams.batch_size)
  # train_x, train_y = create_dataset(train_filenames, batch_size=hparams.batch_size)

  num_train_examples = len(train_filenames)

  # Create the Keras Model
  keras_model = model.create_keras_model(learning_rate=hparams.learning_rate)

  # Setup Learning Rate decay.
  lr_decay = tf.keras.callbacks.LearningRateScheduler(
      lambda epoch: hparams.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
      verbose=True)

  # Train model
  keras_model.fit(
      training_dataset,
      steps_per_epoch=int(num_train_examples / hparams.batch_size),
      epochs=hparams.num_epochs,
      # validation_split=0.1,
      # validation_steps=1,
      verbose=1,
      callbacks=[lr_decay])

  export_path = tf.contrib.saved_model.save_keras_model(
      keras_model, os.path.join(hparams.job_dir, 'keras_export'))
  export_path = export_path.decode('utf-8')
  print('Model exported to: ', export_path)


if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)
  hyperparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hyperparams)
