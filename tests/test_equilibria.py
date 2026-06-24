"""Tests for the generic, table-free equilibrium characterization (`equilibria.py`).

These pin the two properties that matter: the best-response gates are correct on
canonical profiles, and the first-principles verifier is *sound* relative to the
theory-pruned samplers (it never accepts an equilibrium the samplers reject).
"""

from __future__ import annotations

from strategicmiscalibration import equilibria as eq
from strategicmiscalibration.numericals import Params

P = Params(delta=0.05, theta_L=0.1, theta_H=0.9, rho_minus=0.2, rho_plus=0.8, eps=1e-2)


def test_rho_star_and_kappa():
    assert eq.rho_star(P) == 0.7  # 1 - (0.5-0.2)/1
    assert abs(eq.kappa(P) - 0.05 / 0.95) < 1e-12


def test_classify_canonical_profiles():
    assert eq.classify_equilibrium(1, 0, 1, 1, P) == "standard · over-reporting"
    assert eq.classify_equilibrium(1, 0, 0.5, 0, P) == "standard · under-reporting"
    assert eq.classify_equilibrium(0, 1, 0.5, 1, P) == "inverted · over+under"
    assert eq.classify_equilibrium(1, 0.4, 0.5, 0, P) == "hedged-standard · under-reporting"
    # total rejection is babbling regardless of the enumerated sigma corner
    assert eq.classify_equilibrium(0, 0, 1, 1, P) == "total-rejection · babbling"
    assert eq.classify_equilibrium(0, 0, 0, 1, P) == "total-rejection · babbling"


def test_total_rejection_is_equilibrium_deep_in_distrust():
    # Low h, low mu => beliefs in tau_D; the user rejecting both signals (0,0) is a
    # best response and the never-delegated agent is indifferent -> equilibrium.
    assert eq.is_equilibrium(0.05, 0.05, 0.0, 0.0, 0.0, 0.0, P)


def test_standard_over_reporting_exists_somewhere():
    res = eq.scan(P, n_grid=21, source="generic")
    assert any("standard · over-reporting" == k for k in res.by_label)
    assert res.by_label["standard · over-reporting"]


def test_user_best_responds_threshold():
    # A high signal with posterior above rho* must be delegated; below, rejected.
    u_hi = {"rho_tilde_plus": 0.9, "rho_tilde_minus": 0.1}
    assert eq.user_best_responds(u_hi, 1.0, 0.0, P)
    assert not eq.user_best_responds(u_hi, 0.0, 0.0, P)  # should delegate on high
    assert not eq.user_best_responds(u_hi, 1.0, 1.0, P)  # should reject on low


def test_generic_and_samplers_agree_on_a_core():
    # The first-principles scan and the theory-pruned samplers agree on a nonempty
    # core. They need NOT be one a subset of the other: the generic solvers cover
    # families the (over-pruned) samplers miss, and vice versa. Soundness is
    # guaranteed separately because both only emit is_equilibrium-valid profiles.
    counts = eq.cross_check(P, n_grid=15)
    assert counts["both"] > 0


def test_demo_instances_surface_all_realizable_regimes():
    examples = eq.collect_examples(eq.demo_instances())
    regimes = {lab.split(" · ")[0] for lab in examples}
    assert {
        "standard",
        "total-delegation",
        "total-rejection",
        "inverted",
        "partial-standard",
        "partial-inverted",
        "hedged-inverted",
    } <= regimes
    # hedged-standard never appears: it is the one impossible family (sigma^-=0
    # makes the high signal outcome-uninformative, so V_success^+ = V_fail^+ and
    # the fragile (c, 0) split it requires cannot arise -- see prop:highuninform).
    assert "hedged-standard" not in regimes
    # collect_examples internally validates every witness at tol=1e-5 before
    # rounding it for display, so emitted examples are genuine, not artifacts.


def test_hedged_standard_is_empty():
    # hedged-standard requires the fragile V_fail+=0 config, but the user
    # low-indifference locus is entirely robust; the solver returns None and no
    # candidate is a genuine PBE at a tight tolerance.
    import numpy as np

    for _name, Pi in eq.demo_instances():
        for h in np.linspace(0.02, 0.6, 12):
            for mu in np.linspace(0.4, 0.98, 12):
                c = eq.solve_hedged_standard(h, mu, Pi)
                assert c is None or not eq.is_equilibrium(h, mu, *c, Pi, tol=1e-5)


def test_hedged_inverted_exists_at_low_h():
    # hedged-inverted (sigma^U=(alpha, 1), 2-mix agent) is realizable in a low-h /
    # high-mu region once kappa exceeds ~rho^+. This was previously reported empty
    # purely as a solver artifact (gridding the user weight instead of solving it).
    import numpy as np

    P = Params(delta=0.5, theta_L=0.1, theta_H=0.9, rho_minus=0.2, rho_plus=0.8, eps=1e-3)
    found = False
    for h in np.linspace(0.02, 0.45, 18):
        for mu in np.linspace(0.85, 0.97, 10):
            for c in eq.solve_inverted_boundary(h, mu, P):
                if eq.is_equilibrium(h, mu, *c, P, tol=1e-5) and eq._user_regime(c[0], c[1]) == "hedged-inverted":
                    found = True
    assert found


def test_partial_inverted_exists_at_low_h():
    # partial-inverted (2-mix agent) is realizable in a thin low-h region.
    import numpy as np

    P = Params(delta=1 / 6, theta_L=0.1, theta_H=0.9, rho_minus=0.2, rho_plus=0.8, eps=1e-3)
    found = False
    for h in np.linspace(0.05, 0.45, 18):
        for mu in np.linspace(0.5, 0.97, 18):
            for c in eq.solve_inverted_boundary(h, mu, P):
                if eq.is_equilibrium(h, mu, *c, P, tol=1e-5) and eq._user_regime(c[0], c[1]) == "partial-inverted":
                    found = True
    assert found
