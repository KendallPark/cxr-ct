import pydicom
import os
import re
import argparse

def main(args):
  inputFile = args.inFile
  nameparts = re.findall('RI-(.+?)/', inputFile)
  
  readIn = pydicom.read_file(inputFile)
   
  #Print lines
  
  print(nameparts[0])		   #Patient Number
  print(readIn[0x8,0x60].value)    #Modality DX/CT
  try:
    print(readIn[0x18,0x1110].value) #distance from Source to Detector
  except:
    print(' ')

  try:
    print(readIn[0x18,0x1111].value) #distance from Source to Patient
  except:
    print(' ')

  try:
    print(readIn[0x28,0x30].value[0]) #PixelSpacing
  except:
    print(' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inFile',
                        help='input File path',
                        type=str,
                        default='')





args = parser.parse_args()
main(args)

