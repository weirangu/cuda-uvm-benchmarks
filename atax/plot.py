from statistics import mean
from sys import argv
import matplotlib.pyplot as plt
from argparse import ArgumentParser

sizes = ['512', '1024', '2048', '4096', '8192', '16384']
width = 0.2

def create_plot(test_names, title, xlabel, ylabel, filename):
    x = []
    y = []
    xunmanaged = []
    yunmanaged = []
    xticks = []
    for i, test in enumerate(test_names):
        test_file = f"{test}.txt"
        test_file_unmanaged = f"{test}-unmanaged.txt"
        with open(test_file) as f:
            with open(test_file_unmanaged) as uf:
                lines = f.readlines()
                lines_unmanaged = uf.readlines()
                x.append(i)
                xunmanaged.append(i+width)

                managed = mean(map(float, lines))
                unmanaged = mean(map(float, lines_unmanaged))
                y.append(1)
                yunmanaged.append(unmanaged / managed)

                xticks.append(i + width/2)

    plt.bar(x, y, width, label='With UVM')
    plt.bar(xunmanaged, yunmanaged, width, label='Without UVM')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks, test_names)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def create_line_plot(test_name, title, xlabel, ylabel, filename):
    y = []
    yunmanaged = []
    test_file = f"{test_name}.txt"
    test_file_unmanaged = f"{test_name}-unmanaged.txt"
    with open(test_file) as f:
        with open(test_file_unmanaged) as uf:
            lines = f.readlines()
            lines_unmanaged = uf.readlines()

            managed = list(map(float, lines))
            unmanaged = list(map(float, lines_unmanaged))
            y.extend([1] * len(sizes))
            yunmanaged.extend([unmanaged[i] / managed[i] for i in range(len(sizes))])

    plt.plot(sizes, y, 'r', label='With UVM')
    plt.plot(sizes, yunmanaged, 'b', label='Without UVM')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()

if len(argv) < 7:
    print("Pass in 'line'/'bar', name of plot, xlabel, ylabel, output file, and tests to add to the graph. Only provide one test if using line.")
    exit(0)

if argv[1] == 'bar':
    create_plot(argv[6:], argv[2], argv[3], argv[4], argv[5])
else:
    create_line_plot(argv[6], argv[2], argv[3], argv[4], argv[5])
