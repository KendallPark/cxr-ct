from __future__ import print_function
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.filebasedsource import FileBasedSource
from apache_beam.utils import retry
from scipy.ndimage import zoom

import tensorflow as tf
import numpy as np
import sys
import os

import datetime

# from IPython import embed

MIN_CT_VALUE = -1000
MAX_CT_VALUE = 3000

class GCSDirSource(FileBasedSource):
    def read_records(self, file_name, range_tracker):
        yield file_name

class NormalizeNumpy(beam.DoFn):
    def __init__(self, numpy_output_dir=None):
        self._numpy_output_dir = numpy_output_dir

    @retry.with_exponential_backoff(num_retries=5)
    def load_array(self, url):
        fs = beam.io.gcsio.GcsIO()
        return np.load(fs.open(url))

    @retry.with_exponential_backoff(num_retries=5)
    def save_array(self, path, arr):
        fs = beam.io.gcsio.GcsIO()
        file = fs.open(path, 'wb')
        np.save(file, arr)
        file.close()

    def process(self, element):
        path = element
        filename = path.split('/')[-1]
        np_arr = self.load_array(path)

        # cut off min and max values
        np_arr[np_arr < MIN_CT_VALUE] = MIN_CT_VALUE
        np_arr[np_arr > MAX_CT_VALUE] = MAX_CT_VALUE

        assert np.max(np_arr) <= MAX_CT_VALUE, 'values less than max'
        assert np.min(np_arr) >= MIN_CT_VALUE, 'values more than min'

        # make array a cube
        x, y, z = np_arr.shape
        print(np_arr.shape)
        if z < x or z < y:
            startx = (x - z)//2
            starty = (y - z)//2
            cubed_arr = np_arr[startx:startx+z, starty:starty+z, :]
        else:
            max_xyz = max(x, y, z)
            x_pre_pad = (max_xyz - x)//2
            x_post_pad = max_xyz - x - x_pre_pad
            y_pre_pad = (max_xyz - y)//2
            y_post_pad = max_xyz - y - y_pre_pad
            cubed_arr = np.pad(np_arr, ((x_pre_pad, x_post_pad), (y_pre_pad, y_post_pad), (0, 0)), mode='constant', constant_values=MIN_CT_VALUE)
        del np_arr

        x, y, z = cubed_arr.shape
        print(cubed_arr.shape)
        assert x == y, 'x and y dimensions are the same size'
        assert y == z, 'y and z dimensions are the same size'
        assert z == x, 'z and x dimensions are the same size'
        save_path = os.path.join(self._numpy_output_dir, 'normalized_and_padded', filename)

        self.save_array(save_path, cubed_arr)

        scale = 128.0/x
        resized_arr = zoom(cubed_arr, (scale, scale, scale))
        del cubed_arr
        assert resized_arr.shape == (128, 128, 128), 'resized array is 128x128x128'
        save_path = os.path.join(self._numpy_output_dir, 'int-128x128x128', filename)

        self.save_array = self.save_array(save_path, resized_arr)

        yield filename

current_time = datetime.datetime.now().strftime("%Y-%m-%d-at-%Hh%Mm%Ss")

options = PipelineOptions(flags=sys.argv)
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'x-ray-reconstruction'
google_cloud_options.job_name = 'process-numpy-'+current_time
google_cloud_options.staging_location = 'gs://cxr-to-chest-ct2/binaries'
google_cloud_options.temp_location = 'gs://cxr-to-chest-ct2/temp'
# google_cloud_options.machine_type = 'n1-highmem-2'
options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=options) as p:
    input_path = 'gs://cxr-to-chest-ct2/resampled/numpy-int-rotated-correctly/*.npy'
    output_path = 'gs://cxr-to-chest-ct2/volumes/numpy/cubes/'

    numpy_urls = p | 'create urls for np arrays' >> beam.io.Read(GCSDirSource(input_path))
    stuff = numpy_urls | 'normalize data' >> beam.ParDo(NormalizeNumpy(output_path))

    # numpy_urls | 'Print' >> beam.ParDo(lambda (w): print('%s' % (w)))
