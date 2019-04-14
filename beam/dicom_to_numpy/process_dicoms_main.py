from __future__ import print_function
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.filebasedsource import FileBasedSource
from apache_beam.utils import retry
# from google.oauth2 import service_account
import pydicom
import os
import csv
# import io
import tensorflow as tf
import numpy as np
import sys
import gc

import dicom2nifti.patch_pydicom_encodings
dicom2nifti.patch_pydicom_encodings.apply()

from dicom2nifti.convert_dicom import dicom_array_to_nifti
from dicom2nifti.exceptions import ConversionValidationError
from nibabel import processing

# from IPython import embed

# GCS_BUCKET = 'gs://cxr-to-chest-ct/'

class CsvFileSource(FileBasedSource):
  def read_records(self, file_name, range_tracker):
    self._file = self.open_file(file_name)

    reader = csv.reader(self._file)

    for rec in reader:
        yield rec[0], rec[1]

class Split(beam.DoFn):
    def process(self, element):
        key, url = element.split(',')
        yield key, url

class DoEverything(beam.DoFn):
    def __init__(self, gcs_bucket, gcs_save_dir):
        self._gcs_bucket = gcs_bucket
        self._gcs_save_dir = gcs_save_dir

    @retry.with_exponential_backoff(num_retries=5)
    def save_array(self, path, arr):
        fs = beam.io.gcsio.GcsIO()
        file = fs.open(path, 'wb')
        np.save(file, arr)
        file.close()

    def process(self, element):
        key, dir_prefix = element
        fs = beam.io.gcsio.GcsIO()
        gcs_path = os.path.join(self._gcs_bucket, dir_prefix)
        files = fs.list_prefix(gcs_path)
        dicom_paths = [fname for fname in files if '.dcm' in fname]
        dicom_list = [pydicom.dcmread(fs.open(path, 'rb'), defer_size=None, stop_before_pixels=False, force=False) for path in dicom_paths]
        try:
            result = dicom_array_to_nifti(dicom_list, None, False)['NII']
        except ConversionValidationError:
            dicom2nifti.settings.disable_validate_sliceincrement()
            result = dicom_array_to_nifti(dicom_list, None, False)['NII']
            dicom2nifti.settings.enable_validate_sliceincrement()
        series_id = dicom_list[0].SeriesInstanceUID
        del dicom_list
        gc.collect()
        resampled = processing.resample_to_output(result)
        del result
        qoffset = list(resampled.header.get_sform()[:3, -1])
        npy_arr = np.ascontiguousarray(np.flipud(np.rot90(resampled.get_fdata(dtype=np.float32).astype(np.int16))))
        del resampled
        gc.collect()
        path = os.path.join(self._gcs_save_dir, key+'.npy')
        self.save_array(path, npy_arr)
        del npy_arr
        gc.collect()
        yield key, series_id, qoffset, path


class DicomToNifty(beam.DoFn):
    def __init__(self, gcs_bucket):
        self._gcs_bucket = gcs_bucket

    def process(self, element):
        key, dir_prefix = element
        fs = beam.io.gcsio.GcsIO()
        gcs_path = os.path.join(self._gcs_bucket, dir_prefix)
        files = fs.list_prefix(gcs_path)
        dicom_paths = [fname for fname in files if '.dcm' in fname]
        dicom_list = [pydicom.dcmread(fs.open(path, 'rb'), defer_size=None, stop_before_pixels=False, force=False) for path in dicom_paths]
        try:
            result = dicom_array_to_nifti(dicom_list, None, False)
        except ConversionValidationError:
            dicom2nifti.settings.disable_validate_sliceincrement()
            result = dicom_array_to_nifti(dicom_list, None, False)
            dicom2nifti.settings.enable_validate_sliceincrement()
        series_id = dicom_list[0].SeriesInstanceUID
        del dicom_list
        gc.collect()
        yield key, series_id, result['NII']


        # resampled = processing.resample_to_output(result['NII'])
        #
        # t1 = resampled.get_fdata()
        #
        # # Create a 4D Tensor with a dummy dimension for channels
        # t1 = t1[..., np.newaxis]
        #
        # feature = { 'train/series_id': _bytes_feature([series_id]),
        #             'train/volume': _float_feature(t1.reshape(-1)),
        #             'train/volume_shape': _int64_feature(t1.shape),
        #             'train/our_id': _bytes_feature([key]) }
        #
        # example = tf.train.Example(features=tf.train.Features(feature=feature))
        #
        # yield example

class ResampleNifty(beam.DoFn):
    def process(self, element):
        key, series_id, nii = element
        resampled = processing.resample_to_output(nii)
        qoffset = list(resampled.header.get_sform()[:3, -1])
        yield key, series_id, qoffset, resampled

class NiftyToNumpy(beam.DoFn):
    def process(self, element):
        key, series_id, qoffset, nii = element
        yield key, series_id, qoffset, nii.get_fdata(dtype=np.float32).astype(np.int16)

class SaveNumpy(beam.DoFn):
    def __init__(self, gcs_save_dir):
        self._gcs_save_dir = gcs_save_dir

    @retry.with_exponential_backoff(num_retries=5)
    def save_array(self, path, arr):
        fs = beam.io.gcsio.GcsIO()
        file = fs.open(path, 'wb')
        np.save(file, arr)
        file.close()

    def process(self, element):
        key, series_id, qoffset, arr = element
        # fs = beam.io.gcsio.GcsIO()
        path = os.path.join(self._gcs_save_dir, key+'.npy')
        self.save_array(path, arr)
        del arr
        gc.collect()
        yield key, series_id, qoffset

class GenerateTFExamplesFromNumpyEmbeddingsDoFn(beam.DoFn):

    def process(self, element):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        try:
            element = element.element
        except AttributeError:
            pass

        uri, embedding, label_ids = element

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_uri': _bytes_feature([uri]),
            'embedding': _float_feature(embedding.ravel().tolist()),
        }))

        if label_ids:
            label_ids.sort()
            example.features.feature['label'].int64_list.value.extend(label_ids)

        yield example

# class DataflowOptions(PipelineOptions):
#
#   @classmethod
#   def _add_argparse_args(cls, parser):
#     parser.add_argument('--bucket',
#                         help='Input for the pipeline',
#                         default='gs://cxr-to-chest-ct/')
#     parser.add_argument('--output',
#                         help='Output for the pipeline',
#                         default='gs://cxr-to-chest-ct2/resampled/')
#     parser.add_argument('--project',
#                         dest='project',
#                         help='Project',
#                         default='x-ray-reconstruction')
#     parser.add_argument('--temp_location',
#                         dest='temp_location',
#                         help='temp_location',
#                         default='gs://cxr-to-chest-ct2/tmp/')

options = PipelineOptions(flags=sys.argv)
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'x-ray-reconstruction'
google_cloud_options.job_name = 'numpy-highmem-int16-with-rotation'
google_cloud_options.staging_location = 'gs://cxr-to-chest-ct2/binaries'
google_cloud_options.temp_location = 'gs://cxr-to-chest-ct2/temp'
# google_cloud_options.machine_type = 'n1-highmem-2'
options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=options) as p:
    # embed()
    # dicom_urls = p | 'read csv data' >> beam.io.Read(CsvFileSource('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/ct_scan_urls.csv'))

    dicom_urls = p | 'read csv file' >> beam.io.textio.ReadFromText('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/ct_scan_urls.csv') | 'split stuff' >> beam.ParDo(Split())

    saved_arrays = dicom_urls | 'draw the rest of the owl' >> beam.ParDo(DoEverything('gs://cxr-to-chest-ct/', 'gs://cxr-to-chest-ct2/resampled/numpy-int-rotated/'))

    # nii_arrays = dicom_urls | 'dicom to nifty' >> beam.ParDo(DicomToNifty('gs://cxr-to-chest-ct/'))
    # resampled_nii = nii_arrays | 'resample arrays' >> beam.ParDo(ResampleNifty())
    # numpy_arrays = resampled_nii | 'convert to numpy' >> beam.ParDo(NiftyToNumpy())
    # saved_arrays = numpy_arrays | 'save np_arrays' >> beam.ParDo(SaveNumpy('gs://cxr-to-chest-ct2/resampled/numpy-int/'))

    # raw_arrays = dicom_arrays | 'write to bucket' >> beam.io.tfrecordio.WriteToTFRecord('gs://cxr-to-chest-ct2/resampled', coder=beam.coders.ProtoCoder(tf.train.Example), file_name_suffix='.tfrecord.gz')



    # nifti_arrays = dicom_arrays | 'read dicom and convert' >> beam.ParDo(ReadDicom())

    # dicom_arrays | 'Print' >> beam.ParDo(lambda (w, c): print('%s: %s' % (w, c)))
    # dicom_arrays | 'Print' >> beam.ParDo(lambda (w): print('%s' % (w)))

    # dicom.pipeline.run()


    # embed()
    # gcs_path = 'gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/'
    # embed()
    # dicom = p | gcsfilesystem.match([gcs_path])
    # fbs = FileBasedSource('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/*.dcm')
    # dicom = p | fbs.open_file('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/000001.dcm')
    # dicom = p | gcsfilesystem.match('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/*.dcm')
