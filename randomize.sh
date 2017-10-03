#!/bin/bash

##
## Randomizes corrected images
##

FILES=`ls -d *corrected/*.jpg | shuf --random-source=/dev/urandom`
i=0
for f in $FILES; do
	cp "$f" "experiment/$i.jpg"
	DIRNAME=`dirname $f`
	echo "$i $DIRNAME" >> experiment_labels.txt
	((i++))
done

