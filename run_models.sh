#!/bin/bash
##
## Executes the models using all 6 components
##
for X in fixations xy fixationsxy; do
    for y in binary binaryno3; do
	echo "RUNNING $X $y" >> results.txt
	python3 model.py $X $y >> results.txt	
    done
done
