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
import random
import math
import gc
import adjspecies

# from tensorflow.python.ops import image_ops
# from tensorflow_transform.beam.tft_beam_io import transform_fn_io
# from tensorflow_transform.coders import example_proto_coder
# from tensorflow_transform.tf_metadata import dataset_metadata
# from tensorflow_transform.tf_metadata import dataset_schema

# from IPython import embed
# import matplotlib.pyplot as plt

class GCSDirSource(FileBasedSource):
    def read_records(self, file_name, range_tracker):
        yield file_name

class CreateTFExamples(beam.DoFn):

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def compute_drr(self, np_arr, dx=256, dy=256, vx=128, vy=128, vz=128, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0, scd=1800, proj_angle=0, thres=-900):
        import itk
        ImageType = itk.Image[itk.ctype('signed short'), 3]
        dtr = (math.atan(1.0)*4.0)/180.0   # to convert the input angle from degree to radian
        # rotation in x direction with isocenter as center
        thetax = dtr*rx
        thetay = dtr*ry
        thetaz = dtr*rz

        theta_proj_angle = dtr*proj_angle  # Projection angle
        m = 2  # further magnification?
        fac = 1

        img = itk.GetImageFromArray(np_arr)
        img.SetOrigin([0,0,0])

        spacing = np.array(img.GetSpacing()).astype('float32') #voxelsize
        region = img.GetBufferedRegion() # get current image size
        img_size = region.GetSize()

        input_size = np.array(img.GetLargestPossibleRegion().GetSize()).astype('float32')
        iso_c = np.array(img_size)/2
        isocenter = iso_c*spacing

        # apply transformation: Eg: rotation around x-, y-, z-axis
        transformType = itk.Euler3DTransform[itk.D]
        imgtransform = transformType.New()
        imgtransform.SetComputeZYX(True)
        imgtransform.SetTranslation([tx, ty, tz])
        imgtransform.SetRotation(thetax, thetay, thetaz)
        imgtransform.SetCenter(isocenter)

        # set up filter to recover volume
        vol_output_size = np.array([vx, vy, vz])
        vol_size_factor = input_size/vol_output_size
        vol_output_spacing = spacing * vol_size_factor

        resample_image_filter_type = itk.ResampleImageFilter[ImageType, ImageType]
        vol_filter = resample_image_filter_type.New()
        vol_filter.SetInput(img)
        vol_filter.SetDefaultPixelValue(-1000) # air
        vol_filter.SetSize(vol_output_size)
        vol_filter.SetTransform(imgtransform)
        vol_filter.SetOutputSpacing(vol_output_spacing)
        vol_filter.SetOutputOrigin([0,0,0])
        vol_filter.Update()

        vol_output = vol_filter.GetOutput()
        train_volume = itk.GetArrayFromImage(vol_output)


        # image output feature
        img_output_size = [dx, dy, 1]
        img_size_factor = input_size[:1]/(img_output_size[:1]*m)
        img_output_spacing = spacing[:1]*img_size_factor
        img_output_spacing = np.append(img_output_spacing, 1)

        # calculate output values
        # im_sx = spacing[1]/m
        # im_sy = spacing[2]/m
        # Central axis positions are not given by the user. Use the image centers as the central axis position
        o2dx = (dx - 1.0) / 2.0
        o2dy = (dy - 1.0) / 2.0

        # output drr center
        # out_origin = [-im_sx * o2dx, -im_sy * o2dy, -scd]
        # out_spacing = [im_sx, im_sy, 1]
        img_output_origin = img_output_spacing*np.array([-o2dx, -o2dy, -scd])

        # set up interpolator for DRR
        siddon_interpolator_type = itk.TwoProjectionRegistration.SiddonJacobsRayCastInterpolateImageFunction[ImageType, itk.D]
        siddon_interpolator = siddon_interpolator_type.New()
        siddon_interpolator.SetProjectionAngle(theta_proj_angle)
        siddon_interpolator.SetFocalPointToIsocenterDistance(scd)
        siddon_interpolator.SetTransform(imgtransform)

        if thres > -2000:
          siddon_interpolator.SetThreshold(thres)

        siddon_interpolator.Initialize()

        img_filter = resample_image_filter_type.New()
        img_filter.SetInterpolator(siddon_interpolator)
        img_filter.SetInput(img)
        img_filter.SetDefaultPixelValue(-1000)
        img_filter.SetSize(img_output_size)
        img_filter.SetOutputSpacing(img_output_spacing)
        img_filter.SetOutputOrigin(img_output_origin)
        img_filter.Update()

        img_array = itk.GetArrayFromImage(img_filter.GetOutput())

        img_output = img_filter.GetOutput()
        train_image = itk.GetArrayFromImage(img_output)

        return np.squeeze(train_image), train_volume

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
        patient_id, dir_hash = filename.split('-')
        np_arr = self.load_array(path)
        # iterate through all faces of cube
        scd = 1800
        proj_angle = 0
        thres = -900
        tx = 0
        ty = 0
        tz = 0
        rx = 0
        ry = 0
        rz = 0

        np_img, np_vol = self.compute_drr(np_arr, tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz, scd=scd, proj_angle=proj_angle, thres=thres)

        np_img = np_img[..., np.newaxis, np.newaxis]
        np_vol = np_vol[..., np.newaxis]

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': self._int_feature(np_img.ravel()),
            'volume': self._int_feature(np_vol.ravel()),
            'meta/rx': self._float_feature([rx]),
            'meta/ry': self._float_feature([ry]),
            'meta/rz': self._float_feature([rz]),
            'meta/scd': self._float_feature([scd]),
            'meta/proj_angle': self._float_feature([proj_angle]),
            'meta/thres': self._float_feature([thres]),
            'meta/filename': self._bytes_feature([filename.encode("utf8")]),
            'meta/patient_id': self._bytes_feature([patient_id.encode("utf8")]),
            'meta/dir_hash': self._bytes_feature([dir_hash.encode("utf8")])}))
        del np_img
        del np_vol
        gc.collect()
        yield example

session_nickname = adjspecies.random_adjspecies('-')+'-'+datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")

options = PipelineOptions(flags=sys.argv)
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'x-ray-reconstruction'
google_cloud_options.job_name = 'create-tfrecords-'+session_nickname
google_cloud_options.staging_location = 'gs://cxr-to-chest-ct2/binaries'
google_cloud_options.temp_location = 'gs://cxr-to-chest-ct2/temp'
# google_cloud_options.region = 'us-east4'
# google_cloud_options.machine_type = 'n1-highmem-2'
options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=options) as p:
    train_dataset_prefix = os.path.join('gs://cxr-to-chest-ct2/tfrecords/', session_nickname, 'train')
    test_dataset_prefix = os.path.join('gs://cxr-to-chest-ct2/tfrecords/', session_nickname, 'test')

    # path = 'gs://cxr-to-chest-ct2/volumes/numpy/cubes/int-128x128x128/*.npy'
    path = 'gs://cxr-to-chest-ct2/volumes/numpy/cubes/normalized_and_padded/*npy'

    urls = p | 'create urls for np arrays' >> beam.io.Read(GCSDirSource(path))

    train_urls, test_urls = (
        urls
        | 'split dataset' >> beam.Partition(
            lambda elem, _: 1 if elem[-14] == '0' else 0, 2) ) # if the last digit is 0, test set

    # test_urls | 'Print' >> beam.ParDo(lambda (w): print('%s' % (w)))

    _ = (
        train_urls
        | 'create train TFExamples' >> beam.ParDo(CreateTFExamples())
        | 'save as train TFRecords' >> beam.io.WriteToTFRecord(file_path_prefix=train_dataset_prefix, file_name_suffix='.tfrecord', coder=beam.coders.ProtoCoder(tf.train.Example)) )

    _ = (
        test_urls
        | 'create test TFExamples' >> beam.ParDo(CreateTFExamples())
        | 'save as test TFRecords' >> beam.io.WriteToTFRecord(file_path_prefix=test_dataset_prefix, file_name_suffix='.tfrecord', coder=beam.coders.ProtoCoder(tf.train.Example)) )


    # numpy_urls | 'Print' >> beam.ParDo(lambda (w): print('%s' % (w)))
