from subprocess import Popen, PIPE


sizes = [100, 512, 1024, 1536, 2048]

for sz in sizes:
  subprocess.popen("3mm_managed %i >> 3mm_managed.txt"%(sz), shell=True)
  subprocess.popen("3mm %i >> 3mm.txt"%(sz), shell=True)
  





