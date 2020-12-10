rm 3mm.txt
rm 3mm-unmanaged.txt
rm 2DConvolution.txt
rm 2DConvolution-unmanaged.txt

rm 3mm
rm 3mm_man
rm 2DConvolution
rm 2DConvolution-unmanaged

make 3mm
make 3mm_man
make 2DConvolution

for size in 100 512 1024 1536 2048 4096; do
    ./3mm_man $size >> 3mm-managed.txt
    ./3mm $size >> 3mm.txt
    ./2DConvolution $size >> 2DConvo.txt
    ./2DConvolution-unmanaged $size >> 2DConvo-managed.txt
