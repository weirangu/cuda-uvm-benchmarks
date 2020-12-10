from subprocess import Popen


f = open("2DConvolution_tmp.txt", "r")
f2 = open("2DConvolution.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)

f.close()
f2.close()

f = open("2DConvolution-unmanaged_tmp.txt", "r")
f2 = open("2DConvolution-unmanaged.txt", "a")

times = [float(t) for t in f]
avg = sum(times)/5  
a = str(avg) + "\n"
f2.write(a)
