from subprocess import Popen, PIPE


sizes = [100, 512, 1024, 1536, 2048, 4096]

for sz in sizes:
  Popen("./3mm_man %i >> 3mm_managed.txt"%(sz), shell=True)
  Popen("./3mm %i >> 3mm.txt"%(sz), shell=True)
  





