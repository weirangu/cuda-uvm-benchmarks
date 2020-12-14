#!/usr/bin/make


all: bicg_man bicg_unman


bicg_man: bicg_man.cu
		nvcc -g -Wno-deprecated-gpu-targets bicg_man.cu -lcudart -o bicg_man

bicg_unman: bicg_unman.cu
		nvcc -g -Wno-deprecated-gpu-targets bicg_unman.cu -lcudart -o bicg_unman

clean:
	rm -rf bicg_man bicg_unman
