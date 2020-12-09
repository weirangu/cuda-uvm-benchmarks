#!/bin/bash

function run10() {
	echo $2
	for i in {1..10}; do
		$1
	done
}

make clean
make

run10 ./2dconv "2D Convolution Managed"
run10 ./2mm "2mm Managed"
run10 ./3dconv "3D Convolution Managed"

make unmanaged

run10 ./2dconv "2D Convolution Unmanaged"
run10 ./2mm "2mm Unmanaged"
run10 ./3dconv "3D Convolution Unmanaged"

