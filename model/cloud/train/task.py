# Example of how to run.
# gcloud ml-engine jobs submit training my_job_name \
# --module-name train.task --package-path train \
# --staging-bucket 'gs://cxr-to-chest-ct2/gcp-training/staging' \
# --python-version 3.5 --runtime-version 1.13
# --packages packages/Keras-2.2.4.tar.gz,packages/Keras-2.2.4-py3-none-any.whl

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from . import model
# import model  # for local running

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from google.cloud import storage
import keras
from keras.engine.saving import allow_write_to_gcs, allow_read_from_gcs

import datetime
from coolname import generate_slug

# from IPython import embed

session_nickname = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")+'-'+generate_slug(2)

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
      '--session-nickname',
      default=session_nickname,
      type=str,
      help='job session nickname')
  parser.add_argument(
      '--job-dir',
      default=os.path.join('gs://cxr-to-chest-ct2/gcp-training'),
      type=str,
      help='local or GCS location for writing checkpoints and exporting models')
  parser.add_argument(
      '--data-dir',
      # default='tfrecords/silly-cat-2019-04-20--23h57m53s/',
      default='tfrecords/big-dobie-2019-04-21--11h59m35s',
      type=str,
      help='location of the train, eval, and test data, expects train/, eval/, and test/ directories within.')
  parser.add_argument(
      '--bucket',
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
      dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    return dataset

def train_and_evaluate(hparams):

  # Instantiates a client
  storage_client = storage.Client(hparams.project)
  # The name for the bucket
  bucket_name = hparams.bucket
  # Creates the new bucket
  bucket = storage_client.get_bucket(bucket_name)

  train_dir = os.path.join(hparams.data_dir, 'train')
  eval_dir = os.path.join(hparams.data_dir, 'eval')
  train_filenames = [os.path.join('gs://{}/'.format(bucket_name), f.name) for f in bucket.list_blobs(prefix=train_dir) ]
  eval_filenames = [os.path.join('gs://{}/'.format(bucket_name), f.name) for f in bucket.list_blobs(prefix=eval_dir) ]

  num_train_examples = len(train_filenames)
  num_eval_examples = len(eval_filenames)

  training_dataset = create_dataset(train_filenames, batch_size=hparams.batch_size, perform_shuffle=False, shuffle_buffer=num_train_examples)
  validation_dataset = create_dataset(eval_filenames, batch_size=hparams.batch_size)

  # Create the Keras Model
  keras_model = model.create_keras_model(learning_rate=hparams.learning_rate)

  lr_decay = tf.keras.callbacks.LearningRateScheduler(
      lambda epoch: hparams.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
      verbose=True)

  job_path = os.path.join(hparams.job_dir, session_nickname)

  # Adding the callbacks for TensorBoard and Model Checkpoints
  tb_logs_path = os.path.join(job_path, 'logs', 'tensorboard')
  checkpoint_path = os.path.join(job_path, 'checkpoints')

  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tb_logs_path, histogram_freq=0, write_graph=True, write_images=True)
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path)

  # Train model
  keras_model.fit(
      training_dataset,
      steps_per_epoch=int(num_train_examples / hparams.batch_size),
      # steps_per_epoch=1,
      # epochs=1,
      epochs=hparams.num_epochs,
      validation_data=validation_dataset,
      validation_steps=1,
      verbose=1,
      callbacks=[lr_decay, tensorboard, model_checkpoint])

  export_path = tf.contrib.saved_model.save_keras_model(
      keras_model, os.path.join(job_path, 'keras_export'))
  export_path = export_path.decode('utf-8')
  print('Model exported to: ', export_path)


if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)
  hyperparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hyperparams)
