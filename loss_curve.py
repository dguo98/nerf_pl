import numpy as np
import json
import os, sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = sys.argv[1]

    f = open(filepath)
    lines = f.readlines()
    steps = []
    losses = []
    e = 0
    for line in lines:
        d = json.loads(line.rstrip())
        if "epoch_loss" in d:
            losses.append(d["epoch_loss"])
            steps.append(e)
            e += 1
    plt.plot(steps, losses)
    plt.savefig("loss_curve.png")
    f.close()
