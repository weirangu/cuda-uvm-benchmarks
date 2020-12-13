#!/usr/bin/make


atax_man: atax_man.cu
		nvcc -g -Wno-deprecated-gpu-targets atax_man.cu -lcudart -o atax_man

atax_unman: atax_unman.cu
		nvcc -g -Wno-deprecated-gpu-targets atax_unman.cu -lcudart -o atax_unman

clean:
	rm -rf atax_man atax_unman
