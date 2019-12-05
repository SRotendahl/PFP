import re
import numpy as np
import matplotlib.pyplot as plt
import json
import re
def sortJSON(unsort):
    sortedIndices = np.argsort(np.array([int(re.split('(\d+)', data)[1]) for data in unsort]))
    sortedList = [unsort[i] for i in sortedIndices]
    return sortedList

def importJson(jsonF, fut):
    with open(jsonF) as f:
        data = json.load(f)
        dataSets = []
        for key in data[fut]['datasets'].keys():
            dataSets.append(key)
        dataSets = sortJSON(dataSets)
        runTimes = []
        print(data)
        for d in dataSets:
            t = data[fut]['datasets'][d]['runtimes']
            runTimes.append(np.array(t))
        return dataSets, np.array(runTimes)

def plotCpuVsGpu(cpu, gpu, arr):
    avg_gpu = np.mean(gpu, axis = 1)
    avg_cpu = np.mean(cpu, axis = 1)
    speedup = avg_cpu/avg_gpu
    fig, ax1 = plt.subplots()
    p1, = ax1.plot(arr, avg_cpu,label = "No tuning")
    p2, =  ax1.plot(arr, avg_gpu, label = "Tuning")

    plt.xscale('log')
    #ax1.legend(loc="best")
    plt.xlabel("Input size")
    ax1.set_ylabel("Runtime ($\mu$s)")
    plt.title("Own system")

    ax2 = ax1.twinx()
    p3, = ax2.plot(arr, speedup, color = "black", label = "Speed Up")
    ax2.set_ylabel("Speed up")

    plt.legend(handles = [p1, p2, p3], loc="best")

def plotGpuVsGpu(gpu0, gpu1, a1, a2, arr):
    avg_gpu0 = np.mean(gpu0, axis = 1)
    avg_gpu1 = np.mean(gpu1, axis = 1)
    speedup = avg_gpu0/avg_gpu1
    print(avg_gpu0)
    print(avg_gpu1)
    print(speedup)
    fig, ax1 = plt.subplots()
    p1, = ax1.plot(arr, avg_gpu0,label = a1)
    p2, =  ax1.plot(arr, avg_gpu1, label = a2)

    plt.xscale('log')
    plt.xlabel("Input size")
    ax1.set_ylabel("Runtime ($\mu$s)")
    plt.title("Own system")

    ax2 = ax1.twinx()
    p3, = ax2.plot(arr, speedup, color = "black", label = "Speed Up")
    ax2.set_ylabel("Speed up")

    plt.legend(handles = [p1, p2, p3], loc="best")
# Exercise 1

_, notune = importJson("matrix-inversion-no-tune.json", "matrix-inversion.fut")
x1, tune = importJson("matrix-inversion-tune.json", "matrix-inversion.fut")
arr1 = [int(re.split('(\d+)', data)[1]) for data in x1]
plotCpuVsGpu(notune, tune, arr1)
plt.savefig("Task1.png")
