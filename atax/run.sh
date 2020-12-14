rm *.txt
rm *.png

rm atax_man
rm atax_unman

make clean
make atax_man
make atax_unman




for size in 512 1024 2048 4096 8192 16384; do
  rm atax_tmp.txt
  rm atax-unmanaged_tmp.txt

  for i in 1 2 3 4 5; do
    ./atax_man $size >> atax_tmp.txt
    ./atax_unman $size >> atax-unmanaged_tmp.txt
  done
  python3 get_avg.py
done


python3 plot.py line "atax, ratio of run-time vs vector/matrix size" "# of entries in vector/matrix" "ratio of non-UVM compared to UVM" atax_line.png atax 
