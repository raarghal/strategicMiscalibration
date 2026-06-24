"""C3 decision scan: do under-reporting equilibria survive as h -> 0?

Tests the (broken-proof) Contribution-3 claim "under-reporting is impossible with
one-dimensional reputation". Proxy: drive the honesty belief h -> 0 and ask whether
ANY equilibrium with sigma^+ < 1 (easy draw sometimes reported low = under-reporting)
survives, and whether full-sandbagging (sigma^+ = 0) specifically is ruled out.

Run: uv run python scripts/c3_h_to_zero_scan.py
"""

import numpy as np

from strategicmiscalibration.equilibria import (
    classify_equilibrium,
    generic_candidates,
    is_equilibrium,
    rho_star,
)
from strategicmiscalibration.numericals import Params, reputation_updates

CORNER = 1e-6


def is_under(sp):  # easy draw sometimes claims low
    return sp < 1 - CORNER


def is_full_sandbag(sp):
    return sp < CORNER


# delta grid: include the two knife-edges kappa = 1-rho^- and kappa = 1-rho^+,
# plus a spread of generic values. kappa = delta/(1-delta) => delta = k/(1+k).
RHO_M, RHO_P = 0.20, 0.80
k_edges = {"k=1-rho^- (infl)": 1 - RHO_M, "k=1-rho^+ (defl)": 1 - RHO_P}
deltas = {}
for name, k in k_edges.items():
    deltas[name] = k / (1 + k)
for d in (0.02, 0.10, 0.20, 0.30, 0.40, 0.45, 0.49):
    deltas[f"delta={d}"] = d

# h grid pushed hard toward 0; mu full grid.
hs = np.array([1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.49])
mus = np.linspace(0.02, 0.98, 49)

print(f"params: rho^-={RHO_M}, rho^+={RHO_P}, rho^*={rho_star(Params()):.3f}")
print(f"{'delta-arm':22s} {'min-h under-rep':>16s}  {'#under cells':>12s}  {'sigma^+=0 ever?':>15s}")
print("-" * 75)

global_full_sandbag_witnesses = []
for arm, d in deltas.items():
    P = Params(theta_L=0.1, theta_H=0.9, rho_minus=RHO_M, rho_plus=RHO_P, delta=d, eps=1e-3)
    min_h_under = None
    n_under = 0
    full_sandbag_here = []
    under_labels = set()
    for h in hs:
        for mu in mus:
            for cand in generic_candidates(h, float(mu), P, n_2mix=24):
                dp, dm, sp, sm = cand
                if not is_equilibrium(h, float(mu), dp, dm, sp, sm, P):
                    continue
                label = classify_equilibrium(*cand, P)
                if "babbling" in label:
                    continue  # total-rejection: report payoff-irrelevant, sigma^+ arbitrary
                # "shield-active" = genuine sandbagging mechanism: the high report is
                # not always delegated (d_plus<1), so under-reporting can suppress
                # monitoring. Excludes total-delegation reputation-management.
                shield = dp < 1 - CORNER
                if is_under(sp) and shield:
                    n_under += 1
                    under_labels.add(label)
                    if min_h_under is None:
                        min_h_under = h
                    # full sandbag = sigma^+=0 AND high report actually sent on-path
                    # (rho_tilde_plus not NaN). Off-path-supported pooling on low is
                    # excluded so we test genuine on-path full sandbagging.
                    if is_full_sandbag(sp):
                        rt = reputation_updates(h, float(mu), sp, sm, P).get("rho_tilde_plus")
                        onpath = rt is not None and not np.isnan(rt)
                        if onpath:
                            full_sandbag_here.append((round(h, 5), round(float(mu), 3), label))
    global_full_sandbag_witnesses += [(arm, *w) for w in full_sandbag_here]
    mh = f"{min_h_under:.5g}" if min_h_under is not None else "none"
    min_h_fs = min((w[0] for w in full_sandbag_here), default=None)
    fs = f"{len(full_sandbag_here)} (min h={min_h_fs:g})" if full_sandbag_here else "no"
    print(f"{arm:22s} {mh:>16s}  {n_under:>12d}  {fs:>20s}")
    if under_labels:
        print(f"    labels: {sorted(under_labels)}")

print("\n=== full-sandbagging (sigma^+=0) witnesses across all arms ===")
if not global_full_sandbag_witnesses:
    print("NONE — full sandbagging is ruled out everywhere scanned (consistent with audit).")
else:
    for w in global_full_sandbag_witnesses[:40]:
        print("  ", w)
    print(f"  ... total {len(global_full_sandbag_witnesses)}")
