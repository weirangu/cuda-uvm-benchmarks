from subprocess import Popen


f = open("reduction_tmp.txt", "r")
f2 = open("reduction.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)

f.close()
f2.close()

f = open("reduction-unmanaged_tmp.txt", "r")
f2 = open("reduction-unmanaged.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)
