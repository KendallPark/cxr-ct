from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import h5py
import os

def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model-dir',
      default='gs://cxr-to-chest-ct2/gcp-training/vanilla-model/keras_export/1555903332',
      type=str,
      help='project name')
  parser.add_argument(
      '--input-paths',
      default='gs://cxr-to-chest-ct2/tfrecords/silly-cat-2019-04-20--23h57m53s/train-00000-of-00917.tfrecord,gs://cxr-to-chest-ct2/tfrecords/silly-cat-2019-04-20--23h57m53s/train-00001-of-00917.tfrecord',
      type=str,
      help='project name')
  parser.add_argument(
      '--output-dir',
      default='predictions',
      type=str,
      help='project name')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')
  return parser.parse_args()

def _parse_function(proto):
  # define your tfrecord again. Remember that you saved your image as a string.
  keys_to_features = {'image': tf.FixedLenFeature([256, 256, 1], tf.int64),
                      'volume': tf.FixedLenFeature([128 ** 3], tf.int64),
                      'meta/filename': tf.FixedLenFeature([1], tf.string)}

  # Load one example
  parsed_features = tf.parse_single_example(proto, keys_to_features)

  return parsed_features

def create_dataset(filenames):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_function).batch(1)
  return dataset

def main(args):
  test_filenames = args.input_paths.split(',')
  features_dataset = create_dataset(test_filenames)

  filenames = features_dataset.map(lambda features: features['meta/filename'])
  iterator = filenames.make_one_shot_iterator()
  next_filename = iterator.get_next()

  test_dataset = features_dataset.map(lambda features: features['image'] )

  model = tf.contrib.saved_model.load_keras_model(args.model_dir)
  model.summary()
  model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss='mean_squared_error',
    metrics=['accuracy', 'mean_squared_error'])

  with tf.Session() as sess:
    dataset_length = abs(sess.run(tf.data.experimental.cardinality(filenames)))

  predictions = model.predict(test_dataset, steps=dataset_length)

  os.makedirs(args.output_dir, exist_ok=True)

  with tf.Session() as sess:
    for i in range(predictions.shape[0]):
      filename = sess.run(next_filename)[0][0].decode('utf8')
      save_path = os.path.join(args.output_dir, filename.split('.')[0]+'.h5')
      out_cube = predictions[i, :].reshape((128, 128, 128))
      h5f = h5py.File(save_path, 'w')
      h5f.create_dataset('patient_data', data=out_cube)
      h5f.close()

if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)
  main(args)
