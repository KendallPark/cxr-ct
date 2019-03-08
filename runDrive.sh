#!/bin/bash
display_usage(){
        echo -e "\nUsage:  $0 [startIndex] [endIndex]"
        echo -e "\n  Eg: to convert CT (in nifti format) to synthetic Xray or drr for patient 1 to 10 run the line below:" 
        echo -e "\t $0 1 10"
        }
if [ $# -le 1 ]; then
        display_usage
        exit 1
fi
declare -i startI
declare -i endI
startI=${1}
endI=${2}
/home/yingji/testBox

for i in $(seq $startI $endI ); do
	namepart=`printf "%.4d" $i`
	fileLoc=boxRes/${namepart}.nii.gz
	ctlog=`ls CTLogs/log_"$namepart"_*.txt`
	echo $ctlog
	./itkDRR.sh $fileLoc $ctlog
done
