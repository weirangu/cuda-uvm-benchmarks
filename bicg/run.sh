rm *.txt
rm *.png

rm bicg_man
rm bicg_unman

make clean
make bicg_man
make bicg_unman


for size in 512 1024 2048 4096 8192 16384; do
  rm bicg_tmp.txt
  rm bicg-unmanaged_tmp.txt

  for i in 1 2 3 4 5; do
    ./bicg_man $size >>   bicg_tmp.txt
    ./bicg_unman $size >> bicg-unmanaged_tmp.txt
  done

  python3 get_avg.py
done


python3 plot.py line "bicg, ratio of run-time vs vector/matrix size" "# of entries in vector/matrix" "ratio of non-UVM compared to UVM" bicg_line.png bicg 
