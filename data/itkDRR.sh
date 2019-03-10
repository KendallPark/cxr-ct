#!/bin/bash
display_usage(){
        echo -e "\nUsage:  $0 [nifti file path] [CT log file from extractNifti.sh]"
        echo -e "\n  Eg: " 
        echo -e "\t $0 /home/user/patient1.nii.gz log_patient1.txt"
        }
if [ $# -le 1 ]; then
        display_usage
        exit 1
fi

workingDir="/home/yingji/testBox"

inputNii=${1}

#rawInfo=`fslinfo "$inputNii"`
#dim1=`echo $rawInfo | cut -d " " -f4`
#dim3=`echo $rawInfo | cut -d " " -f8`

ctlogfile=${2}
dim1=`awk 'NR==6 {print $1}' "$ctlogfile"`
dim3=`awk 'NR==6 {print $3}' "$ctlogfile"`

niiFileName=`basename $inputNii`
subjID=${niiFileName%%.nii*}

logFile=${workingDir}/$subjID-drrlog.txt
outputPath=${workingDir}/imageDRR/$subjID-drr.tif

ctInfo=${workingDir}/CTLogs/log_"$subjID"_*.txt
dxInfo=${workingDir}/DXLogs/DX_log_"$subjID"_*.txt

scpDist=`awk 'NR==4' $dxInfo`
#Get DX distance
if [[ $scpDist == ' ' ]]; then 
	scpDist=$(( 1000 + (RANDOM % 5) )) #randomly add 0-5 mm to 1000mm, not sure if this is necessary or not
#else
	#scpDist=$(( $scpDist + (RANDOM % 2 - 1) ))
fi


pix=`awk 'NR==5' $ctInfo`

proj=`seq 0 .01 5 | shuf | head -n1`

# SET rotation for x, y, z, axes: 
#rx=`seq 0 .01 1 | shuf | head -n1`
#ry=`seq 0 .01 1 | shuf | head -n1`
#rz=`seq 0 .01 1 | shuf | head -n1`
rx=0;
ry=0
rz=0

# Set isocenter location
ix=$((${dim1}/2))
iy=${ix}
iz=`printf "%.0f" $(echo "scale=2;${dim3}/2" | bc)`
#ix=$(( ix + (RANDOM % 5 - 2) ))
#iy=$(( iy + (RANDOM % 5 - 2) ))
#iz=$(( iz + (RANDOM % 5 - 2) ))

#ix=0
#iy=0
#iz=0

reso=0.1

# since i'm setting the resolution pixel to be 0.1mm, the output image sizes should be multiple by the following factor
factor=`echo "$pix/$reso" | bc`
imgW=$(( ( dim1*factor ) + 50 ))
imgH=$(( ( dim3*factor ) + 50 )) 

#echo $imgW

/software/ITK-build/bin/TwoProjectionRegistrationTestDriver getDRRSiddonJacobsRayTracing -v -rp $proj -rx $rx -ry $ry -rz $rz -iso $ix $iy $iz -res 0.1 0.1 -size $imgW $imgH -scd $scpDist -threshold -1000 -o $outputPath $inputNii >> $logFile

