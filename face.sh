#!/bin/bash

## Face recognition bash script for looping thorugh a folder
## Thanks to: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

DIR=$1
OUT=$2

for file in $DIR/*.jpg; do
	name=$(basename "$file")
	i=0
	python3 face-alignment/align_faces.py \
		--shape-predictor "face-alignment/shape_predictor_68_face_landmarks.dat" \
		--image $file \
		--out "$2/${name%.*}_${i}.${name##*.}"
done
