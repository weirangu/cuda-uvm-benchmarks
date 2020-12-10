rm *.txt
rm *.png

rm 3mm
rm 3mm_man
rm 2DConvolution
rm 2DConvolution-unmanaged

make 3mm
make 3mm_man
make 2DConvolution


for i in 100 512 1024 1536 2048 4096; do
    ./3mm_man 200 >> 3mm.txt
    ./3mm 200 >> 3mm-unmanaged.txt
    ./2DConvolution 4096 >> 2DConvolution.txt
    ./2DConvolution-unmanaged 4096 >> 2DConvolution-unmanaged.txt
done

python3 plot.py bar performance tests time bar.png 2DConvolution 3mm

rm *.txt

for size in 50 100 200 250 512 1024; do
    ./3mm_man $size >> 3mm.txt
    ./3mm $size >> 3mm-unmanaged.txt
done


for size in 50 100 200 250 512 1024; do
    for i in 1 2 3 4 5; do
        ./2DConvolution $size >> 2DConvolution_tmp.txt
        ./2DConvolution-unmanaged $size >> 2DConvolution-unmanaged_tmp.txt
    done
    python3 holyfuckthisissostupid.py
done

python3 plot.py line 2D_line tests time 2D_line.png 2DConvolution 
python3 plot.py line 3mm_line tests time 3mm_line.png 3mm 
