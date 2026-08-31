import numpy as np
import pytest

exec(open("src/functions/function_defs.py").read())


def test_get_neighbors():
    L = 6

    neel = 2 * ((np.indices((L, L)).sum(axis=0)) % 2) - 1

    spinsup = get_neighbors(neel, (0,0))
    print(spinsup)
    assert (spinsup == np.array([1,1,1,1])).all() #comparison is done element wise and .all() checks if all are true

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