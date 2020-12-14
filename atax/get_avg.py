from subprocess import Popen


f = open("atax_tmp.txt", "r")
f2 = open("atax.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)

f.close()
f2.close()

f = open("atax-unmanaged_tmp.txt", "r")
f2 = open("atax-unmanaged.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)
