import dipy
from dipy.align.reslice import reslice
from dipy.data import get_fnames

import dicom2nifti
import nibabel as nib
import re
import os
import argparse

def main(args):
  inputfolder = args.inDir #eg: /tmp/patient0001/ct
  outputfolder = args.out #eg: /tmp/patient0001/nifti
  logfile = args.logFile
  nameparts = re.findall('RI-(.+?)/',inputfolder) #should work most of the time...
  out_uninterp = os.path.join(outputfolder, nameparts[0] + "raw.nii.gz")
  outputpath = os.path.join(outputfolder, nameparts[0] + ".nii.gz")

  dicom2nifti.dicom_series_to_nifti(inputfolder, out_uninterp, reorient_nifti=False)
  nib.openers.Opener.default_compresslevel = 9
  
  # now out_uninterp is in ls /tmp/patient0001/testnifti
  nif_img = nib.load(out_uninterp)

  #print('Original volume dimension:',nif_img.shape)
  data = nif_img.get_data()
  affine = nif_img.affine
  zooms = nif_img.header.get_zooms()[:3]
  #print('Original voxel dimension:',zooms)

  voxelsize = nif_img.header.get_zooms()[0]
  new_zooms = (voxelsize, voxelsize, voxelsize)
  # Reslice to get isotropic volume
  data2, affine2 = reslice(data, affine, zooms, new_zooms)
  #print('Volume dimension after reslice:',data2.shape)
  img_new = nib.Nifti1Image(data2, affine2)
  nib.save(img_new,outputpath)
  f = open(logfile,"a+")
  f.write("%d %d %d\n" % img_new.shape)
  f.close() 

  #print('Voxel dimension after reslice:', img_new.header.get_zooms()[:3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inDir',
                        help='input folder path',
                        type=str,
                        default='')
    parser.add_argument('--out',
                        help='output file path',
                        type=str,
                        default='')
    parser.add_argument('--logFile',
                        help='CT log file path',
                        type=str,
                        default='')

args = parser.parse_args()
main(args)
