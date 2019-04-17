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
# import itk
# import datetime
# import random

# from tensorflow_transform.beam.tft_beam_io import transform_fn_io
# from tensorflow_transform.coders import example_proto_coder
# from tensorflow_transform.tf_metadata import dataset_metadata
# from tensorflow_transform.tf_metadata import dataset_schema

from IPython import embed

class GCSDirSource(FileBasedSource):
    def read_records(self, file_name, range_tracker):
        yield file_name

class TFExampleFromImageDoFn(beam.DoFn):
    def __init__(self, numpy_output_dir=None, image_output_dir=None):
        self._numpy_output_dir = numpy_output_dir
        self._image_output_dir = image_output_dir

def _bytes_feature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(self, value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(self, value):
    return tf.train.Feature(int16_list=tf.train.Int16List(value=value))


def compute_drr(self, np_arr, rx=0, ry=0, rz=0):
    dtr = (math.atan(1.0)*4.0)/180.0   # to convert the input angle from degree to radian
    # rotation in x direction with isocenter as center
    thetax = dtr*rx
    thetay = dtr*ry
    thetaz = dtr*rz

    thres = -900  # Threshold value
    proj_angle = dtr*8  # Projection angle
    scd = 941  # distance from source to patient, value from x-ray above
    m = 2  #further magnification?
    fac = 1

    img = itk.GetImageFromArray(np_arr)
    img.SetOrigin([0,0,0])
    spacing = img.GetSpacing() #voxelsize
    region = img.GetBufferedRegion() # get current image size

    imgSize = region.GetSize()

    inputSize = img.GetLargestPossibleRegion().GetSize()
    isoC = np.array([imgSize[0]/2, imgSize[1]/2, imgSize[2]/2])
    isocenter = spacing[1]*isoC

    dx = int((imgSize[0]*m)*fac)-50
    dy = int((imgSize[2]*m)*fac)-50

    im_sx = spacing[1]/m
    im_sy = spacing[1]/m
    o2dx = (dx - 1.0) / 2.0
    o2dy = (dy - 1.0) / 2.0

    #output image center
    outOrigin = [-im_sx * o2dx, -im_sy * o2dy, -scd]
    outspacing = [im_sx, im_sy, 1]

    rotation_center = np.array([0, 0, 0])

    # apply transformation: Eg: rotation around x-, y-, z-axis
    transformType = itk.Euler3DTransform[itk.D]
    imgtransform = transformType.New()
    imgtransform.SetComputeZYX(True)
    imgtransform.SetRotation(thetax, thetay, thetaz)
    imgtransform.SetCenter(isocenter)

    filterType = itk.ResampleImageFilter[ImageType, ImageType]
    imgfilter = filterType.New()
    imgfilter.SetInput(img)
    imgfilter.SetDefaultPixelValue(0)
    imgfilter.SetSize(inputSize)

    # Setup Interpolator
    interpolatorType = itk.TwoProjectionRegistration.SiddonJacobsRayCastInterpolateImageFunction[ImageType, itk.D]
    interpolator = interpolatorType.New()
    interpolator.SetProjectionAngle(proj_angle)
    interpolator.SetFocalPointToIsocenterDistance(scd)
    interpolator.SetTransform(imgtransform)

    if thres > -2000:
      interpolator.SetThreshold(thres)

    interpolator.Initialize()

    imgfilter.SetInterpolator(interpolator)
    imgfilter.SetSize([dx, dy, 1])
    imgfilter.SetOutputSpacing(outspacing)
    imgfilter.SetOutputOrigin(outOrigin)
    imgfilter.Update()

    img_array = itk.GetArrayFromImage(imgfilter.GetOutput())

    return np.squeeze(img_array)

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
        np_arr = self.load_array(element)
        # reassign CT tube to smallest value
        non_cyl_min = np.min(np_arr[np_arr > -2000])
        np_arr[np_arr < -1500] = non_cyl_min

        # make array a cube
        x, y, z = np_arr.shape
        if z < min(x, y):
            startx = x//2 - z//2
            starty = y//2 - z//2
            cubed_arr = np_arr[startx:startx+z, starty:starty+z, :]
        else:
            max_xyz = max(x, y, z)
            startx = (max_xyz - x)//2
            endx = max_xyz - x - startx
            starty = (max_xyz - y)//2
            endy = max_xyz - y - starty
            cubed_arr = np.pad(np_arr, ((startx, endx), (starty, endy), (0, 0)), mode='constant', constant_values=non_cyl_min)
        del np_arr

        x, y, z = cubed_arr.shape
        assert x == y and y == z and z == x, 'all dimensions are the same size'

        scale = 128.0/x
        resized_arr = zoom(cubed_arr, (scale, scale, scale))
        del cubed_arr
        assert resized_arr.shape == (128, 128, 128), 'resized array is 128x128x128'


options = PipelineOptions(flags=sys.argv)
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'x-ray-reconstruction'
google_cloud_options.job_name = 'process-training-data'
google_cloud_options.staging_location = 'gs://cxr-to-chest-ct2/binaries'
google_cloud_options.temp_location = 'gs://cxr-to-chest-ct2/temp'
# google_cloud_options.machine_type = 'n1-highmem-2'
options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=options) as p:
    import random
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H.%M.%S")

    train_dataset_prefix = os.path.join('gs://cxr-to-chest-ct2/tfrecords/', current_time, 'train')
    test_dataset_prefix = os.path.join('gs://cxr-to-chest-ct2/tfrecords/', current_time, 'test')
    train_drr_prefix = os.path.join('gs://cxr-to-chest-ct2/drr/', current_time, 'train')
    test_drr_prefix = os.path.join('gs://cxr-to-chest-ct2/drr/', current_time, 'test')
    test_percent = 20.0
    random.seed(0)

    # gcs = beam.io.gcp.gcsfilesystem.GCSFileSystem(options)
    # coder = example_proto_coder.ExampleProtoCoder(metadata.schema)

    path = 'gs://cxr-to-chest-ct2/resampled/numpy-int-rotated/*.npy'

    # urls = list(map(lambda x: x.path, gcs.match(['gs://cxr-to-chest-ct2/resampled/numpy-int-rotated/*.npy'])[0].metadata_list))

    # numpy_urls = p | 'create urls for np arrays' >> beam.Create(urls)

    numpy_urls = p | 'create urls for np arrays' >> beam.io.Read(GCSDirSource(path))

    # assert 0 < test_percent < 100, 'test_percent must in the range (0-100)'

    # train_dataset, test_dataset = (
    #     numpy_urls
    #     | 'Split dataset' >> beam.Partition(
    #         lambda elem, _: int(random.uniform(0, 100) < test_percent), 2))

    # embed()

    stuff = numpy_urls | 'create training tf_records' >> beam.ParDo(TFExampleFromImageDoFn())

    # _ = (
    #     train_dataset
    #     | 'Write train dataset' >> tfrecordio.WriteToTFRecord(
    #         train_dataset_prefix, coder))
    #
    # eval_dataset_prefix = os.path.join(eval_dataset_dir, 'part')
    # _ = (
    #     eval_dataset
    #     | 'Write eval dataset' >> tfrecordio.WriteToTFRecord(
    #         eval_dataset_prefix, coder))


    # numpy_urls = p | 'get urls for np arrays' >> beam.ParDo(GetNumpyUrls('gs://cxr-to-chest-ct2/resampled/numpy-int-rotated/*.npy'))

    # numpy_urls | 'Print' >> beam.ParDo(lambda (w): print('%s' % (w)))

    # embed()
    # dicom_urls = p | 'read csv data' >> beam.io.Read(CsvFileSource('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/ct_scan_urls.csv'))

    # dicom_urls = p | 'read csv file' >> beam.io.textio.ReadFromText('gs://cxr-to-chest-ct/datasets/LIDC-IDRI Dataset/ct_scan_urls.csv') | 'split stuff' >> beam.ParDo(Split())
