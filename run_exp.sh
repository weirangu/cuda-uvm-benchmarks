rm *.txt
rm *.png

rm 3mm
rm 3mm_man
rm 2DConvolution
rm 2DConvolution-unmanaged

make clean
make 3mm
make 3mm_man
make 2DConvolution
make add_man
make add_unman


#for i in 100 512 1024 1536 2048 4096; do
#    ./3mm_man 200 >> 3mm.txt
#    ./3mm 200 >> 3mm-unmanaged.txt
#    ./2DConvolution 4096 >> 2DConvolution.txt
#    ./2DConvolution-unmanaged 4096 >> 2DConvolution-unmanaged.txt
#done
#
#python3 plot.py bar performance tests time bar.png 2DConvolution 3mm
#
#rm *.txt

for size in 200 250 512 1024 2048 4096; do
    ./3mm_man $size >> 3mm.txt
    ./3mm $size >> 3mm-unmanaged.txt
done


for size in 50 100 200 250 512 1024; do
    rm 2DConvolution_tmp.txt
    rm 2DConvolution-unmanaged_tmp.txt
    for i in 1 2 3 4 5; do
        ./2DConvolution $size >> 2DConvolution_tmp.txt
        ./2DConvolution-unmanaged $size >> 2DConvolution-unmanaged_tmp.txt
    done
    python3 get_avg.py
done


python3 plot.py line "2D convolution, ratio of run-time with diff sizes" size ratio 2D_line.png 2DConvolution 
python3 plot.py line "3 matrix multiplication, ratio of run-time with diff sizes " size ratio 3mm_line.png 3mm 


#for i in 18 19 20 21 22 23; do
#    ./add_man $i >> add.txt
#    ./add_unman $i >> add-unmanaged.txt
#done
#
#python3 plot.py line add "size, 2^x" ratio add.png add

#./add_man >> add.txt
#./add_unman >> add-unmanaged.txt
#
#python3 plot.py line add iteration time add.png add
#
#./3mm_man 1024 >> 3mm.txt
#./3mm 1024 >> 3mm-unmanaged.txt
#
#
#python3 plot.py line 3_matrix_mult_by_iteration iteration time 3mm_it.png 3mm
