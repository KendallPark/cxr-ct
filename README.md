## python and ITK-Based workflow to generate pseudo-Xray

If interested to run the scripts here, change all directories to your local directories

This set of scripts will need python module (1)dicom2nifit, (2)dipy, (3)pydicom, (4)nibbabel
and ITK to work

To convert series of dicom images into 3D, interpolated, nifti volume, run:
	./extractNifti.sh <Start index> <End index> (note: index are patient id)

To create synthetic xray  / DRR from the CT volume geerated above, run:
	./runDrive.sh <Start index> <End index> 



