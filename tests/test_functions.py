import numpy as np

from ising_mc.functions.function_defs import _attempt_metropolis_update
from ising_mc.functions.function_defs import get_neighbors
from ising_mc.functions.function_defs import sum_neighbors
from ising_mc.functions.function_defs import H
from ising_mc.functions.function_defs import M
from ising_mc.functions.function_defs import dH
from ising_mc.functions.function_defs import metropolis_step
from ising_mc.functions.function_defs import run_metropolis

import pytest

def test_get_neighbors():
    L = 6

    neel = 2 * ((np.indices((L, L)).sum(axis=0)) % 2) - 1

    spinsup = get_neighbors(neel, (0,0))
    print(spinsup)
    assert (spinsup == np.array([1,1,1,1])).all() #comparison is done element wise and .all() checks if all are true


def test_sum_neighbors():
    L = 6
    neel = 2 * ((np.indices((L, L)).sum(axis=0)) % 2) - 1

    assert sum_neighbors(neel, (0, 0)) == 4

#this parametrization only ever applies to the next testfunction
@pytest.mark.parametrize("J,h,expected_H", 
                          [
                              (1, 0, -2*36),
                              (-1, 0, 2*36),
                              (0, 1, 0)
                          ],
                          )

def test_H(J,h,expected_H):
    L = 6
    
    neel = 2 * ((np.indices((L, L)).sum(axis=0)) % 2) - 1

    out = H(neel, J, h)

    assert out == expected_H

@pytest.mark.parametrize("J,h,expected_dH", 
                          [
                              (1, 0, 8),
                              (-1, 0, -8),
                              (0, 1, 2)
                          ],
                          )

def test_dH(J, h, expected_dH):

    L = 6
    
    neel = 2 * ((np.indices((L, L)).sum(axis=0)) % 2) - 1

    out = dH(neel, (0,0), J, h)

    assert out == expected_dH

def test_metropolis_step_accept(monkeypatch):
    state = np.ones((4, 4), dtype=int)

    positions = iter([1, 2])

    monkeypatch.setattr(
        np.random,
        "randint",
        lambda s: next(positions),
    )

    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda: 0.0002,
    )

    #dH will return 8 here
    #thus we have comp = exp(-1 * 8) = 0.00033...

    result = metropolis_step(state, beta=1, J=-1, h=0) 

    #we therefore expect the spin to be flipped and all other so be left unflipped
    assert result[1, 2] == -1
    assert np.sum(result == -1) == 1


def test_metropolis_step_accepts_energy_decrease_without_random_draw(monkeypatch):
    L = 4
    state = 2 * ((np.indices((L, L)).sum(axis=0)) % 2) - 1
    positions = iter([0, 0])

    monkeypatch.setattr(np.random, "randint", lambda s: next(positions))

    def fail_if_called():
        pytest.fail("Energy-decreasing updates should not draw a random value")

    monkeypatch.setattr(np.random, "uniform", fail_if_called)

    result = metropolis_step(state, beta=1, J=-1, h=0)

    assert result[0, 0] == 1


def test_accepted_update_reports_observable_changes(monkeypatch):
    state = np.ones((4, 4), dtype=int)
    initial_energy = H(state, J=-1, h=0)
    initial_magnetization = M(state)
    positions = iter([1, 2])

    monkeypatch.setattr(np.random, "randint", lambda s: next(positions))
    monkeypatch.setattr(np.random, "uniform", lambda: 0.0002)

    delta_energy, delta_magnetization = _attempt_metropolis_update(
        state, beta=1, J=-1, h=0
    )

    assert initial_energy + delta_energy == H(state, J=-1, h=0)
    assert initial_magnetization + delta_magnetization == M(state)


def test_run_metropolis_does_not_return_configurations_by_default():
    result = run_metropolis(
        L=2,
        beta=1,
        J=-1,
        h=0,
        num_samples=1,
        burn_in_steps=0,
        steps_between_samples=0,
    )

    assert len(result) == 6


def test_run_metropolis_returns_sampled_configurations(monkeypatch):
    initial_state = np.ones((2, 2), dtype=int)
    positions = iter([(0, 0), (0, 1), (1, 0)])

    monkeypatch.setattr(
        "ising_mc.functions.function_defs.random_config",
        lambda L: initial_state.copy(),
    )

    def flip_next_spin(state, beta, J, h):
        pos = next(positions)
        magnetization_change = -2 * state[pos]
        state[pos] *= -1
        return 0, magnetization_change

    monkeypatch.setattr(
        "ising_mc.functions.function_defs._attempt_metropolis_update",
        flip_next_spin,
    )

    result = run_metropolis(
        L=2,
        beta=1,
        J=0,
        h=0,
        num_samples=3,
        burn_in_steps=0,
        steps_between_samples=1,
        return_configurations=True,
    )

    sampled_configurations = result[-1]
    expected_configurations = np.array(
        [
            [[1, 1], [1, 1]],
            [[-1, 1], [1, 1]],
            [[-1, -1], [1, 1]],
        ]
    )

    assert len(result) == 7
    np.testing.assert_array_equal(sampled_configurations, expected_configurations)
