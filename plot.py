from statistics import mean
from sys import argv
import matplotlib.pyplot as plt
from argparse import ArgumentParser

colours = ['r', 'g', 'b']
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

                y.append(mean(map(float, lines)))
                yunmanaged.append(mean(map(float, lines_unmanaged)))

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

if len(argv) < 6:
    print("Pass in name of plot, xlabel, ylabel, output file, and tests to add to the graph")
    exit(0)
create_plot(argv[5:], argv[1], argv[2], argv[3], argv[4])