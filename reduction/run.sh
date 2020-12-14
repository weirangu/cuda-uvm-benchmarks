rm *.txt
rm *.png

make clean
make reduction


for size in 128 256 512 1024 2048; do
  rm reduction_tmp.txt
  rm reduction-unmanaged_tmp.txt

  for i in 1 2 3 4 5; do
    ./reduction $size >> reduction_tmp.txt
    ./reduction-unmanaged $size >> reduction-unmanaged_tmp.txt
  done
  python3 get_avg.py
done


python3 plot.py line "reduction, ratio of run-time vs vector/matrix size" "# of entries in vector/matrix" "ratio of non-UVM compared to UVM" reduction_line.png reduction 
