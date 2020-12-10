from subprocess import Popen, PIPE


sizes = [100, 512, 1024, 1536, 2048, 4096]

#for sz in sizes:

for i in range(5):
  Popen("./3mm_man 1024 >> 3mm.txt", shell=True)
  Popen("./3mm 1024 >> 3mm-unmanaged.txt", shell=True)
  Popen("./2DConvolution 15000 >> 2DConvolution.txt", shell=True)
  Popen("./2DConvolution-unmanaged 15000 >> 2DConvolution-unmanaged.txt", shell=True)

  





