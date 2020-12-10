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

python3 experiment.py

