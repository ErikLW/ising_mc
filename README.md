#ising_mc.jl

The main function `run_metropolis()` of `ising_mc.jl` runs a
metropolis monte-carlo calculation of the classical Ising model.
It returns thermodynamic expectation values.
It can be used to return the samples, that were used for the expectation values. They are used to train neural networks to recognize the phase structure from samples in [machinelearn_phases.jl](https://github.com/ELW/machinelearn_phases.jl)