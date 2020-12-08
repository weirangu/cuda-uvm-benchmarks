#!/bin/bash

make clean
make
echo "2D Convolution Managed"
for i in {1..5}; do
	./2dconv
done

make unmanaged
echo "2D Convolution Unmanaged"
for i in {1..5}; do
	./2dconv
done

