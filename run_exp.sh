rm *.txt

rm 3mm
rm 3mm_man
rm 2DConvolution
rm 2DConvolution-unmanaged

make 3mm
make 3mm_man
make 2DConvolution


#for size in 100 512 1024 1536 2048 4096; do
for i in 100 512 1024 1536 2048 4096; do
    ./3mm_man 1024 >> 3mm.txt
    ./3mm 1024 >> 3mm-unmanaged.txt
    ./2DConvolution 4096 >> 2DConvolution.txt
    ./2DConvolution-unmanaged 4096 >> 2DConvolution-unmanaged.txt
done

python3 plot.py bar please tests time bar.png 2DConvolution 3mm


for size in 100 512 1024 1536 2048 4096; do
    ./3mm_man $size >> 3mm.txt
    ./3mm $size >> 3mm-unmanaged.txt
    ./2DConvolution $size >> 2DConvolution.txt
    ./2DConvolution-unmanaged $size >> 2DConvolution-unmanaged.txt
done


python3 plot.py line 2D_line tests time 2D_line.png 2DConvolution 
python3 plot.py line 3mm_line tests time 3mm_line.png 3mm 
