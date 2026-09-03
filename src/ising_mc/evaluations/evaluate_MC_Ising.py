import numpy as np
import ising_mc as ising
import matplotlib.pyplot as plt
import pandas as pd

temperatures = [i for i in np.linspace(1,3,20)]
print(temperatures)
L = 6

for i, T in enumerate(temperatures):
    Temp, e, evar, m, mvar, absm = ising.run_metropolis(
        L=L,
        beta=1/T,
        J=-1,
        h=1e-2,
        num_samples=3000,
        burn_in_steps=2*(10**3)*(L**2),
        steps_between_samples=10*(L**2),
    )

    row = pd.DataFrame([{"Linear_size": L,
                     "Temperature": Temp,
                     "Energy_per_site": e,
                     "Energy_variance": evar,
                     "Magnetization": m,
                     "Magnetization_variance": mvar,
                     "Abs_Magnetization": absm}])

    row.to_csv(
        "../../../data/results_h1e-2_test.csv",
        mode = "a",
        header = (i == 0),
        index = False
    )
