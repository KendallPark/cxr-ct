#!/bin/bash
display_usage(){
	echo -e "\nUsage:  $0 [startIndex] [endIndex]"
	echo -e "\n  Eg: to convert dicom to nifti for patient 1 to 10 run the line below:" 
	echo -e "\t $0 1 10"
	}
if [ $# -le 1 ]; then
	display_usage
	exit 1
fi

declare -i count
declare -i startI
declare -i endI
startI=${1}
endI=${2}

targetLoc="/home/yingji/BoxTest"
workingDir="/home/yingji/testBox"

outputLoc="$workingDir/boxRes"
ctlogDir="$workingDir/CTLogs"
dxlogDir="$workingDir/DXLogs"

repoDir="$workingDir/xray_itk"

if [[ ! -d $outputLoc ]]; then
	mkdir $outputLoc
fi
if [[ ! -d $ctlogDir ]]; then
	mkdir $ctlogDir
fi
if [[ ! -d $dxlogDir ]]; then
	mkdir $dxlogDir
fi

for i in $(seq $startI $endI ); do
  namepart=`printf "%.4d" $i`
  fileLoc=${targetLoc}/LIDC-IDRI-"$namepart"

  cd $fileLoc

  targetFile=`find "$PWD" -name "000001.dcm"`
  outlist=`ls $targetFile -1 | wc`
  outcount=`echo "$outlist" | awk '{ print $1;}'` #number of files... 
  count=0
  if [[ "$outcount" -ge 1 ]]; then
	while IFS=" " read -ra arr; do
		
		for j in "${arr[@]}"; do
			(( count += 1 ))

			echo $j
			echo "$count"	
			if [[ -f "$j" ]]; then
				targetDir=$(dirname "$j")

				valueLog="$workingDir/log_${namepart}_${count}.txt"				

				python ${repoDir}/getDicomInfo.py --inFile $j >> $valueLog
				ct_val=`awk 'NR==2' $valueLog`
				stp_val=`awk 'NR==4' $valueLog`


				if [[ "$ct_val" == "CT" ]]; then
					python3 ${repoDir}/preproc_nifti.py --inDir "${targetDir}" --out "${outputLoc}" --logFile "$valueLog" 
					mv "$valueLog" "$workingDir/CTLogs/"
				else
					mv $valueLog ${valueLog/log/DX_log}
				fi
				
			fi
		done
	done <<< "${targetFile}"
  fi

done

cd $workingDir
mv DX_log* $dxlogDir

