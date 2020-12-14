from subprocess import Popen


f = open("bicg_tmp.txt", "r")
f2 = open("bicg.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)

f.close()
f2.close()

f = open("bicg-unmanaged_tmp.txt", "r")
f2 = open("bicg-unmanaged.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)
