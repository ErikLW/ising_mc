def random_config(L):
    """Generate a random spin configuration on an ``L x L`` square lattice.

    Each lattice site is assigned a spin independently and uniformly from
    ``{-1, +1}``.

    Parameters
    ----------
    L : int
        Linear system size. The returned configuration contains ``L**2``
        spins.

    Returns
    -------
    numpy.ndarray
        Two-dimensional array of shape ``(L, L)`` containing spins ``-1``
        and ``+1``.
    """
    #return np.random.randint(0,2,(L,L))
    return np.random.choice([-1, 1], (L,L))


def get_neighbors(state, pos):
    """Return the four nearest-neighbor spins of a lattice site.

    Neighbors are taken in the positive and negative directions along both
    lattice axes. Periodic boundary conditions are applied, so sites at one
    edge of the lattice are adjacent to sites at the opposite edge.

    Parameters
    ----------
    state : numpy.ndarray
        Two-dimensional spin configuration with shape ``(L, L)`` and entries
        ``-1`` or ``+1``.
    pos : tuple of int
        Lattice position ``(i, j)`` whose nearest neighbors are requested.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the four neighboring spins.
    """
    L, _ = state.shape
    i, j = pos
    neighbors = np.array([state[((i+1) % L,j % L)], state[((i-1) % L,j % L)], state[(i % L,(j + 1) % L)], state[(i % L,(j - 1) % L)]])
    return neighbors


def H(state, J, h):
    """Compute the Ising energy of a spin configuration.

    The energy is evaluated for nearest-neighbor interactions on a square
    lattice with periodic boundary conditions, using the sign convention
    implemented in this function.

    Parameters
    ----------
    state : numpy.ndarray
        Two-dimensional spin configuration with shape ``(L, L)`` and entries
        ``-1`` or ``+1``.
    J : float
        Coupling strength between nearest-neighbor spins.
    h : float
        External magnetic-field strength.

    Returns
    -------
    float
        Total energy of the configuration.
    """
    L, _ = state.shape
    energy_h = h * state.sum()
    state_shift_x = np.roll(state, 1, axis=0)
    state_shift_y = np.roll(state, 1, axis=1)

    energy_J_x = np.sum(J * state * state_shift_x)
    energy_J_y = np.sum(J * state * state_shift_y)

    return energy_h + energy_J_x + energy_J_y


def M(state):
    """Compute the total magnetization of a spin configuration.

    Parameters
    ----------
    state : numpy.ndarray
        Two-dimensional spin configuration with entries ``-1`` or ``+1``.

    Returns
    -------
    int or numpy.integer
        Total magnetization, equal to the sum of all spins in ``state``.
    """
    return np.sum(state)


def dH(state, pos, J, h):
    """Compute the energy change associated with flipping one spin.

    The returned value is the change in the energy implemented by :func:`H`
    when the spin at ``pos`` is replaced by its negative. Nearest neighbors
    are evaluated with periodic boundary conditions.

    Parameters
    ----------
    state : numpy.ndarray
        Two-dimensional spin configuration with shape ``(L, L)`` and entries
        ``-1`` or ``+1``.
    pos : tuple of int
        Position ``(i, j)`` of the spin proposed for flipping.
    J : float
        Coupling strength between nearest-neighbor spins.
    h : float
        External magnetic-field strength.

    Returns
    -------
    float
        Energy difference ``H(flipped_state) - H(state)`` for the proposed
        single-spin flip.
    """
    dh = - 2 * h * state[pos] 
    dJ = - 2 * J * state[pos] * np.sum(get_neighbors(state,pos))
    return dh + dJ


def metropolis_step(state,beta, J, h):
    """Perform one single-spin Metropolis update.

    A lattice site is selected uniformly at random. The corresponding spin is
    flipped with the Metropolis acceptance rule based on the energy change
    returned by :func:`dH`. If the move is accepted, ``state`` is modified in
    place.

    Parameters
    ----------
    state : numpy.ndarray
        Two-dimensional spin configuration with shape ``(L, L)`` and entries
        ``-1`` or ``+1``.
    beta : float
        Inverse temperature used in the Metropolis acceptance probability.
    J : float
        Coupling strength between nearest-neighbor spins.
    h : float
        External magnetic-field strength.

    Returns
    -------
    numpy.ndarray
        The updated spin configuration. This is the same array object as the
        input ``state``, with one spin flipped if the proposal was accepted.
    """
    L, _ = state.shape
    pos = tuple(np.random.randint(s) for s in state.shape)#random tuple from 0:L-1
    r = np.random.uniform()
    comp = np.exp(-beta * dH(state, pos, J, h))
    if r < comp:
        state[pos] *= -1
        return state
    else:
        return state


def run_metropolis(L, beta, J, h, N, init_sweeps, sample_interval):
    """Run a Metropolis Monte Carlo simulation of the square-lattice Ising model.

    The simulation starts from a random spin configuration. It first performs
    ``init_sweeps`` single-spin Metropolis steps as burn-in. It then records
    ``N`` samples of the total energy and magnetization, performing
    ``sample_interval`` single-spin Metropolis steps between successive
    recorded samples.

    Parameters
    ----------
    L : int
        Linear system size. The simulated lattice has shape ``(L, L)`` and
        contains ``L**2`` spins.
    beta : float
        Inverse temperature of the simulation.
    J : float
        Coupling strength between nearest-neighbor spins.
    h : float
        External magnetic-field strength.
    N : int
        Number of samples used to evaluate the reported observables.
    init_sweeps : int
        Number of initial single-spin Metropolis steps discarded as burn-in
        before sampling begins.
    sample_interval : int
        Number of single-spin Metropolis steps performed between successive
        recorded samples.

    Returns
    -------
    temperature : float
        Temperature corresponding to the supplied inverse temperature,
        ``1 / beta``.
    energy_per_site : float
        Mean sampled energy divided by the number of lattice sites.
    energy_var : float
        Variance of the sampled total energy divided by the square of the
        number of lattice sites.
    mag_per_site : float
        Mean sampled magnetization divided by the number of lattice sites.
    mag_var : float
        Variance of the sampled total magnetization divided by the square of
        the number of lattice sites.
    abs_mag_per_site : float
        Absolute value of the mean magnetization per site.
    """
    state = random_config(L)

    for i in range(init_sweeps):
        state = metropolis_step(state, beta, J, h)

    H_arr = np.array([])
    H_var_arr = np.array([])
    M_arr = np.array([])
    M_var_arr = np.array([])

    for i in range(N):
        H_arr = np.append(H_arr, H(state, J, h))
        M_arr = np.append(M_arr, M(state))

        for i in range(sample_interval):
            state = metropolis_step(state, beta, J, h)

    energy = np.mean(H_arr)
    energy_per_site = energy / (state.shape[0] * state.shape[1])

    energy_var = np.var(H_arr) / ((state.shape[0] * state.shape[1])**2)


    mag = np.mean(M_arr)
    mag_per_site = mag / (state.shape[0] * state.shape[1])
    mag_var = np.var(M_arr) / ((state.shape[0] * state.shape[1])**2)

    absmag = (np.abs(M_arr)).mean()
    
    return 1/beta, energy_per_site, energy_var, mag_per_site, mag_var, absmag
