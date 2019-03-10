## python and ITK-Based workflow to generate pseudo-Xray

If interested to run the scripts here, change all directories to your local directories (for all bash script.. i believe...)

This set of scripts will need python modules: (1)dicom2nifit, (2)dipy, (3)pydicom, (4)nibbabel
and ITK (only needed for creating synthetic xray) 


To convert series of dicom images into 3D, interpolated, nifti volume, run:

	./extractNifti.sh <Start index> <End index> (note: index are patient id)


To create synthetic xray  / DRR from the CT volume generated above, run:

	./runDrive.sh <Start index> <End index> 



Building ITK : 
(quick build ITK: https://itk.org/Wiki/ITK/Getting_Started/Build/Linux ; complete software guide of ITK: https://itk.org/ItkSoftwareGuide.pdf)
