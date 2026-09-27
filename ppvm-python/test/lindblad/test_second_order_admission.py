# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""`pc_step_arr(..., admission="second_order")`: the admitted set is the
top-room by the second-order end-of-step weight `dt·(L*x) + ½dt²·(L*²x)`
computed densely, the uncapped step has an O(dt³) local error, and the
default predictor-corrector path is unchanged.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ppvm import Lindbladian
from ppvm.lindblad import _basis_to_codes

from ._helpers import all_strings, coo_to_dense, expm_mv_dense

N = 4
FULL = all_strings(N)
FULL_INDEX = {s: i for i, s in enumerate(FULL)}


def _random_model(rng):
    """Random two-body couplings (no ties among the weights) plus Z dephasing."""
    h_terms = []
    for a in range(N):
        for b in range(a + 1, N):
            for pa in "XYZ":
                for pb in "XYZ":
                    if rng.random() < 0.3:
                        s = ["I"] * N
                        s[a], s[b] = pa, pb
                        h_terms.append(("".join(s), float(rng.normal())))
    for a in range(N):
        s = ["I"] * N
        s[a] = "XZ"[a % 2]
        h_terms.append(("".join(s), float(rng.normal())))
    jumps = [("I" * j + "Z" + "I" * (N - j - 1), 0.3) for j in range(N)]
    return Lindbladian(N, h_terms, jumps)


def _dense_generator(lind):
    codes = _basis_to_codes(FULL, N)
    return coo_to_dense(lind.generator_arr(codes), len(FULL))


def _to_strings(codes):
    letters = np.array(list("IXZY"))  # ppvm codes: 0=I, 1=X, 2=Z, 3=Y
    return ["".join(letters[row]) for row in codes]


def _full_vector(strings, coeffs):
    v = np.zeros(len(FULL))
    for s, c in zip(strings, coeffs):
        v[FULL_INDEX[s]] = c
    return v


def _dense_second_order_weights(M, basis, x, dt, keep_first=None):
    """`w_Q` over strings outside `basis`, and the first-generation set.

    `keep_first` emulates the candidate-map cap of pass 1: only the
    `keep_first` largest first-generation strings carry `dt·ℓ_Q` and act as
    parents (all maps here fit in one accumulation chunk, so the emulation
    is exact).
    """
    in_basis = np.array([s in set(basis) for s in FULL])
    y = M @ _full_vector(basis, x)
    first = [i for i in range(len(FULL)) if not in_basis[i] and y[i] != 0.0]
    if keep_first is not None:
        first = sorted(first, key=lambda i: -abs(y[i]))[:keep_first]
    parents = np.where(in_basis, y, 0.0)
    parents[first] = y[first]
    w = 0.5 * dt**2 * (M @ parents)
    w[first] += dt * y[first]
    w[in_basis] = 0.0
    return w, {FULL[i] for i in first}


@pytest.mark.parametrize("slack", [None, 1.0])
def test_admitted_set_is_dense_top_room(slack):
    rng = np.random.default_rng(7)
    lind = _random_model(rng)
    M = _dense_generator(lind)
    dt = 0.2
    basis = ["XZII", "IYXI", "ZIIY", "IIZX", "YIXI", "IXIZ"]
    x = rng.normal(size=len(basis))
    room = 12
    # cap_map_to_room keeps room + 1 entries (no ties here).
    keep = None if slack is None else room + 1
    w, first = _dense_second_order_weights(M, basis, x, dt, keep_first=keep)
    outside = [i for i in range(len(FULL)) if w[i] != 0.0]
    assert len(outside) > room
    top = [FULL[i] for i in sorted(outside, key=lambda i: -abs(w[i]))[:room]]
    expected_second = sum(s not in first for s in top)
    assert expected_second > 0

    A = len(basis) + room
    b, _, info = lind.pc_step_arr_timed(
        _basis_to_codes(basis, N),
        x,
        dt,
        max_basis=A,
        drop_tol=0.0,
        admit_basis=A,
        admission="second_order",
        candidate_slack=slack,
    )
    assert set(_to_strings(b)) == set(basis) | set(top)
    assert info["admitted1"] == room
    assert info["admitted2"] == 0
    assert info["n_second_gen_admitted"] == expected_second


def test_uncapped_step_is_third_order():
    rng = np.random.default_rng(11)
    lind = _random_model(rng)
    M = _dense_generator(lind)
    basis = ["ZIII", "IXYI"]
    x = np.array([1.0, 0.5])
    x_full = _full_vector(basis, x)
    errors = []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        b, c = lind.pc_step_arr(
            _basis_to_codes(basis, N),
            x,
            dt,
            max_basis=10_000,
            drop_tol=0.0,
            admission="second_order",
        )
        exact = expm_mv_dense(dt * M, x_full)
        errors.append(np.linalg.norm(_full_vector(_to_strings(b), c) - exact))
    for prev, curr in pairwise(errors):
        assert curr < prev / 6, f"second-order admission not O(dt^3): {errors}"


def test_default_admission_unchanged():
    rng = np.random.default_rng(3)
    lind = _random_model(rng)
    basis = _basis_to_codes(["ZIII", "IZII", "XXII"], N)
    x = np.array([1.0, -0.4, 0.2])
    kw = dict(dt=0.1, max_basis=20, drop_tol=0.0, admit_basis=40)
    b0, c0 = lind.pc_step_arr(basis, x, **kw)
    b1, c1 = lind.pc_step_arr(basis, x, admission="pc", **kw)
    assert np.array_equal(b0, b1)
    assert np.array_equal(c0, c1)


@pytest.mark.parametrize(
    "kw",
    [dict(tau_add=1e-3), dict(candidate_slack=0.5), dict()],
)
def test_invalid_second_order_config(kw):
    lind = _random_model(np.random.default_rng(0))
    basis = _basis_to_codes(["ZIII"], N)
    args = (basis, np.array([1.0]), 0.1, 10)
    if not kw:
        with pytest.raises(ValueError):
            lind.pc_step_arr(*args, admission="third_order")
    else:
        with pytest.raises(Exception, match="invalid step configuration"):
            lind.pc_step_arr(*args, admission="second_order", **kw)


def test_pc_replace_matches_dense_emulation():
    """First admission top-room by |ℓ|; predictor on S ∪ F; the second
    admission keeps the top-room of F by |x_pred| and the new candidates by
    ½dt·|ℓ′|; corrector from the pre-step state on the result."""
    rng = np.random.default_rng(5)
    lind = _random_model(rng)
    M = _dense_generator(lind)
    dt = 0.3
    basis = ["XZII", "IYXI", "ZIIY", "IIZX", "YIXI", "IXIZ"]
    x = rng.normal(size=len(basis))
    room = 10
    S = [FULL_INDEX[s] for s in basis]
    x_full = _full_vector(basis, x)

    y = M @ x_full
    cand = [i for i in range(len(FULL)) if i not in S and y[i] != 0.0]
    F = sorted(cand, key=lambda i: -abs(y[i]))[:room]
    W1 = S + F
    x_pred = np.zeros(len(FULL))
    x_pred[W1] = expm_mv_dense(dt * M[np.ix_(W1, W1)], x_full[W1])
    lp = M @ x_pred
    G = [i for i in range(len(FULL)) if i not in W1 and lp[i] != 0.0]
    pool = [(abs(x_pred[i]), i) for i in F] + [(0.5 * dt * abs(lp[i]), i) for i in G]
    top = [i for _, i in sorted(pool, key=lambda p: -p[0])[:room]]
    n_new = sum(i in G for i in top)
    assert 0 < n_new < room
    W2 = S + top
    x_corr = expm_mv_dense(dt * M[np.ix_(W2, W2)], x_full[W2])

    A = len(basis) + room
    b, c, info = lind.pc_step_arr_timed(
        _basis_to_codes(basis, N),
        x,
        dt,
        max_basis=A,
        drop_tol=0.0,
        admit_basis=A,
        admission="pc_replace",
    )
    got = dict(zip(_to_strings(b), c))
    assert set(got) == {FULL[i] for i in W2}
    np.testing.assert_allclose([got[FULL[i]] for i in W2], x_corr, atol=1e-10)
    assert info["admitted1"] == room
    assert info["admitted2"] == n_new
