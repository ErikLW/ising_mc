import numpy as np
import ising_mc as ising
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../data/results_h1e-2.csv")

plt.plot(df["Temperature"], df["Abs_Magnetization"], marker = "o")

plt.xlabel("Temperature")
plt.ylabel("Absolute magnetization")

plt.savefig("magnetization.png", dpi=300, bbox_inches="tight")

plt.show()

