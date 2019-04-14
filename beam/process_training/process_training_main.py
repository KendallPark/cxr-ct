from __future__ import print_function
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.utils import retry

import tensorflow as tf
import numpy as np

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



options = PipelineOptions(flags=sys.argv)
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'x-ray-reconstruction'
google_cloud_options.job_name = 'process-training-data'
google_cloud_options.staging_location = 'gs://cxr-to-chest-ct2/binaries'
google_cloud_options.temp_location = 'gs://cxr-to-chest-ct2/temp'
# google_cloud_options.machine_type = 'n1-highmem-2'
options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=options) as p:
    # embed()
    # dicom_urls = p | 'read csv data' >> beam.io.Read(CsvFileSource('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/ct_scan_urls.csv'))

    dicom_urls = p | 'read csv file' >> beam.io.textio.ReadFromText('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/ct_scan_urls.csv') | 'split stuff' >> beam.ParDo(Split())
