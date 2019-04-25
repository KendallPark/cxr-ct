from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
from task import create_dataset
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

def main(args):
  model = tf.contrib.saved_model.load_keras_model(args.model_dir)
  model.summary()
  model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss='mean_squared_error',
    metrics=['accuracy', 'mean_squared_error'])

  test_filenames = args.input_paths.split(',')
  test_dataset = create_dataset(test_filenames)
  predictions = model.predict(test_dataset, steps=len(test_filenames))

  os.makedirs(args.output_dir, exist_ok=True)

  for i in range(len(test_filenames)):
    save_path = os.path.join(args.output_dir, test_filenames[i].split('/')[-1].split('.')[0]+'.h5')
    out_cube = predictions[i, :].reshape((128, 128, 128))
    h5f = h5py.File(save_path, 'w')
    h5f.create_dataset('patient_data', data=out_cube)
    h5f.close()

if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)
  main(args)
