#!/usr/bin/make

EXES = 2DConvolution 2mm 3DConvolution reduction

all: $(EXES) transferManaged transfer

%:  %.cu
	nvcc -g -Wno-deprecated-gpu-targets $^ -lcudart -o $@
	nvcc -g -Wno-deprecated-gpu-targets -DUNMANAGED $^ -lcudart -o $@-unmanaged

transferManaged: transferBandwidthManaged.cu
		nvcc -g -Wno-deprecated-gpu-targets transferBandwidthManaged.cu -lcudart -o transferManaged

transfer: transferBandwidth.cu
		nvcc -g -Wno-deprecated-gpu-targets transferBandwidth.cu -lcudart -o transfer

3mm: 3mm.cu
		nvcc -g -Wno-deprecated-gpu-targets 3mm.cu -lcudart -o 3mm

3mm_man: 3mm_managed.cu
		nvcc -g -Wno-deprecated-gpu-targets 3mm_managed.cu -lcudart -o 3mm_man

atax: atax.cu
		nvcc -g -Wno-deprecated-gpu-targets atax.cu -lcudart -o atax

bicg: bicg.cu
		nvcc -g -Wno-deprecated-gpu-targets bicg.cu -lcudart -o bicg

add_man: add_man.cu
		nvcc -g -Wno-deprecated-gpu-targets add_man.cu -lcudart -o add_man

add_unman: add_unman.cu
		nvcc -g -Wno-deprecated-gpu-targets add_unman.cu -lcudart -o add_unman

clean:
	rm -rf $(EXES) *-unmanaged transferManaged transfer
