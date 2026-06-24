"""Generic, table-free equilibrium characterization for the monopolistic game.

This is the redesign of the equilibrium scan. Instead of transcribing the
appendix tables of the paper as ~30 hand-coded ``cond_T*R*`` / sampler / row
specs (fragile, error-prone, and already out of sync between ``main.tex`` and
``revised_main.tex``), it characterizes equilibria **from first principles**:

* a candidate ``(sigma^U, sigma^A)`` at beliefs ``(h, mu)`` is accepted iff BOTH
  players best-respond (``is_equilibrium``), using the verified Bayes engine
  ``reputation_updates`` and the incentive helpers from :mod:`numericals`;
* each accepted equilibrium is given a **descriptive** label computed from its
  realized strategies (``classify_equilibrium``) — e.g. ``"hedged-standard ·
  under-reporting"`` — so figures read directly rather than referentially.

Avoiding the combinatorial blow-up (see the design note in the chat): we never
grid ``sigma`` and ``d`` blindly.

* **Pure equilibria** are found by checking the 16 corner profiles
  ``d, sigma in {0,1}^2`` against the best-response gates — exact and grid-free.
* **Mixed (boundary-user) equilibria** live on indifference loci and are solved
  by root-finds, not grids. One-mix families (partial-standard, hedged-standard)
  are pairs of 1-D root-finds; the two-mix inverted-favoring families
  (partial-inverted, hedged-inverted) need the agent mixing on both signals
  (``solve_inverted_boundary``). The mixing weights are *pinned* by indifference.
* The original **samplers are kept as an optional fast path**
  (``source="samplers"``): they are the theory-pruned closed forms, so they cost
  no runtime solving and they carry exact mixing weights. Run through the SAME
  gates + labeler, a sampler bug now costs only coverage (caught by
  ``cross_check``), never soundness.
* A blunt ``coarse_fallback`` grid exists for completeness but is OFF by default
  — it is exactly the unpruned enumeration we want to avoid.

Reuses the verified primitives from :mod:`numericals` (Bayes updates, trust
regions, ``agent_exp_payoff``, the ``agent_best_responds`` IC gate).

Realizability (all 8 user-strategy cases are *solved*; not all are *non-empty*)
------------------------------------------------------------------------------
With the payoff parameters fixed (``c, e, r`` constant; here ``rho^* = 0.7``) and
only ``delta, theta_L, theta_H, rho^-, rho^+`` varied, a scan over the belief
plane realizes **7 of the 8** user-strategy families:

* **Realizable**: standard, total-delegation, total-rejection, inverted,
  partial-standard (substantial regions), partial-inverted (a thin low-``h``
  region; needs the 2-mix agent), and hedged-inverted (a low-``h`` / high-``mu``
  region for ``kappa`` above roughly ``rho^+``; also needs the 2-mix solve).
* **Empty / impossible**: only **hedged-standard**. The fragile high-side config
  it requires, ``(V_success^+, V_fail^+) = (c, 0)`` (\\cref{app:hedged_standard_del}),
  cannot arise. Its own agent profile has ``sigma^-=0`` (no low-side
  over-reporting), so a high report is sent *only* on genuinely easy tasks; the
  delegation outcome is then a ``Bernoulli(rho^+)`` draw independent of the
  agent's type, leaving the post-high success/failure posteriors identical. Hence
  ``V_success^+ = V_fail^+`` for *every* hedged-standard profile, contradicting
  the required ``c != 0`` split. This is the proof appended to ``revised_main.tex``
  (``prop:highuninform``, ``cor:nohs``, ``prop:nounder``, ``rem:hsgap``); it voids
  ``lem:hsband`` and the deflation branch of ``thm:phase`` in the 2-type model.
  (Earlier this module also flagged hedged-inverted as empty — that was a solver
  gap: it gridded ``sigma^-`` and the user weight and hoped the IC gate caught a
  hit, but a mixed equilibrium needs the user weight to *exactly* zero the agent's
  indifference. Solving for it instead (:func:`solve_inverted_boundary`) surfaces
  hedged-inverted readily. ``sigma^-=0`` does not arise there, so the argument
  above does not touch it.)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .numericals import (
    Params,
    aep_high_minus_low,
    agent_best_responds,
    reputation_updates,
)

CORNER = 1e-6  # tolerance for "is this strategy coordinate at a corner (0/1)?"


def rho_star(P: Params) -> float:
    return 1.0 - (P.e - P.c) / P.r


def kappa(P: Params) -> float:
    return P.delta / (1.0 - P.delta)


# ============================================================
# Generic best-response gates  (replace all cond_T*R* / user_ic_*)
# ============================================================


def user_best_responds(u: dict, d_plus: float, d_minus: float, P: Params, tol=1e-6) -> bool:
    """The user delegates iff its posterior success prob exceeds ``rho_star``.

    For each signal: if ``rho_tilde > rho_star`` it must delegate (d=1); if below,
    reject (d=0); at indifference any ``d`` is a best response. A signal sent with
    zero probability (``rho_tilde`` is NaN) is off-path, so any ``d`` is allowed.
    """
    rs = rho_star(P)

    def ok(rho_tilde, d):
        if rho_tilde is None or np.isnan(rho_tilde):
            return True
        if rho_tilde > rs + tol:
            return d >= 1.0 - 1e-6
        if rho_tilde < rs - tol:
            return d <= 1e-6
        return True

    return ok(u.get("rho_tilde_plus"), d_plus) and ok(u.get("rho_tilde_minus"), d_minus)


def is_equilibrium(
    h: float,
    mu: float,
    d_plus: float,
    d_minus: float,
    sigma_plus: float,
    sigma_minus: float,
    P: Params,
    tol=1e-3,
) -> bool:
    """A profile is a PBE iff both players best-respond (beliefs are Bayes-consistent
    by construction in ``reputation_updates``)."""
    u = reputation_updates(h, mu, sigma_plus, sigma_minus, P)
    return user_best_responds(u, d_plus, d_minus, P) and agent_best_responds(
        u, sigma_plus, sigma_minus, d_plus, d_minus, P, ic_tol=tol
    )


# ============================================================
# Descriptive classification  (replaces the T#R# / table labels)
# ============================================================


def _user_regime(d_plus: float, d_minus: float) -> str:
    hi = d_plus > 1 - CORNER
    lo_p = d_plus < CORNER
    hi_m = d_minus > 1 - CORNER
    lo_m = d_minus < CORNER
    if hi and lo_m:
        return "standard"
    if lo_p and hi_m:
        return "inverted"
    if hi and hi_m:
        return "total-delegation"
    if lo_p and lo_m:
        return "total-rejection"
    if hi and not (hi_m or lo_m):
        return "hedged-standard"
    if lo_m and not (hi or lo_p):
        return "partial-standard"
    if lo_p and not (hi_m or lo_m):
        return "partial-inverted"
    if hi_m and not (hi or lo_p):
        return "hedged-inverted"
    return "interior-user"


def _agent_behavior(sigma_plus: float, sigma_minus: float) -> str:
    # Honest reporting is sigma^+ = 1 (easy -> high), sigma^- = 0 (hard -> low).
    over = sigma_minus > CORNER  # hard type sometimes claims high
    under = sigma_plus < 1 - CORNER  # easy type sometimes claims low
    if over and under:
        return "over+under"
    if over:
        return "over-reporting"
    if under:
        return "under-reporting"
    return "honest"


def classify_equilibrium(d_plus, d_minus, sigma_plus, sigma_minus, P: Params) -> str:
    """Descriptive label, e.g. ``'hedged-standard · under-reporting'``.

    Under total rejection the agent is never delegated and never monitored, so its
    report is payoff-irrelevant (any ``sigma`` is a best response); we label that
    ``babbling`` rather than reading an over/under direction off an arbitrary
    enumerated corner.
    """
    regime = _user_regime(d_plus, d_minus)
    if regime == "total-rejection":
        return f"{regime} · babbling"
    return f"{regime} · {_agent_behavior(sigma_plus, sigma_minus)}"


# ============================================================
# Candidate generation
# ============================================================

Candidate = Tuple[float, float, float, float]  # (d_plus, d_minus, sigma_plus, sigma_minus)


def _bisect(f: Callable[[float], float], lo=0.0, hi=1.0, iters=60) -> Optional[float]:
    """Return a root of monotone ``f`` in [lo, hi], or None if no sign change."""
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if np.isnan(fmid):
            return None
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def _rho_tilde(h, mu, sp, sm, P, high):
    rt = reputation_updates(h, mu, sp, sm, P).get("rho_tilde_plus" if high else "rho_tilde_minus")
    return np.nan if rt is None else rt


def solve_hedged_standard(h: float, mu: float, P: Params) -> Optional[Candidate]:
    """Hedged-standard ``sigma^U=(1, beta)``, agent ``sigma^A=(sigma^+<1, 0)`` (the
    theory's only agent profile here; \\cref{app:hedged_standard_del}).

    User low-indifference pins ``sigma^+`` (``rho_tilde(rho^-)=rho^*``), agent
    indifference at ``rho^+`` pins ``beta``. **This family is empty in the 2-type
    model** (see the module docstring): the theory requires ``V_fail^+=0`` but the
    low-indifference locus is entirely robust (``V_fail^+=c``), so no interior
    ``beta`` exists. Kept for completeness; returns ``None`` in practice.
    """
    rs = rho_star(P)
    sp = _bisect(lambda s: _rho_tilde(h, mu, s, 0.0, P, high=False) - rs)
    if sp is None or not (CORNER < sp < 1 - CORNER):
        return None
    u = reputation_updates(h, mu, sp, 0.0, P)
    beta = _bisect(lambda b: aep_high_minus_low(u, P.rho_plus, 1.0, b, P))
    if beta is None or not (CORNER < beta < 1 - CORNER):
        return None
    return (1.0, beta, sp, 0.0)


def solve_partial_standard(h: float, mu: float, P: Params) -> Optional[Candidate]:
    """Partial-standard ``sigma^U=(q, 0)``, agent ``sigma^A=(1, sigma^->0)``
    (over-reporting; \\cref{app:partial_standard_del}).

    User high-indifference pins ``sigma^-`` (``rho_tilde(rho^+)=rho^*``) by diluting
    the high pool with hard over-reporters; the hard type's indifference pins ``q``.
    """
    rs = rho_star(P)
    sm = _bisect(lambda s: _rho_tilde(h, mu, 1.0, s, P, high=True) - rs)
    if sm is None or not (CORNER < sm < 1 - CORNER):
        return None
    u = reputation_updates(h, mu, 1.0, sm, P)
    q = _bisect(lambda a: aep_high_minus_low(u, P.rho_minus, a, 0.0, P))
    if q is None or not (CORNER < q < 1 - CORNER):
        return None
    return (q, 0.0, 1.0, sm)


def solve_inverted_boundary(h: float, mu: float, P: Params, n=12) -> List[Candidate]:
    """The two **inverted-favoring boundary** families, whose agent reports
    *invertedly* — it under-reports easy draws and over-reports hard draws,
    ``sigma^A=(sigma^+<1, sigma^->0)`` (\\cref{app:partial_inverted_del},
    \\cref{app:hedged_inverted_del}):

    * partial-inverted ``sigma^U=(0, beta)`` — user rejects high, mixes low;
      user-indifference is on the **low** signal, ``rho_tilde(rho^-)=rho^*``.
    * hedged-inverted ``sigma^U=(alpha, 1)`` — user delegates low, mixes high;
      user-indifference is on the **high** signal, ``rho_tilde(rho^+)=rho^*``.

    Each is genuinely *two-mix*: the user mixes on one signal and the agent mixes
    on one of its own coordinates. The earlier implementation gridded ``sigma^-``
    *and* the user weight and relied on the IC gate to catch a hit — but a mixed
    equilibrium requires the user weight to **exactly** zero the agent's
    indifference, which a coarse grid essentially never lands (this is why
    hedged-inverted came up spuriously empty). We instead solve it as nested 1-D
    root-finds, with the agent pinned to a corner on its non-mixed coordinate:

    1. the agent-mixed coordinate (``sigma^+`` or ``sigma^-``) is solved from the
       user-indifference locus ``rho_tilde(.)=rho^*`` (root-find in that
       coordinate);
    2. the user mixing weight is then solved from the agent's indifference at the
       mixed coordinate — ``aep_high_minus_low`` is *affine* in the delegation
       probability, so this is a clean 1-D root-find;
    3. the best-response gate validates the corner coordinate's IC, the user's IC
       on the forced signal, and Bayes-consistency.

    Two agent sub-cases cover the family: ``mix_high`` (``sigma^-=1`` corner,
    ``sigma^+`` interior) and ``mix_low`` (``sigma^+=0`` corner, ``sigma^-``
    interior); both yield the over+under inverted profile. Pruned to ``h < 1/2``
    (\\cref{prop:noinv}: no inverted-favoring equilibrium when ``h >= 1/2``). ``n``
    is retained for API compatibility and is unused. Both families are non-empty
    in a thin low-``h`` strip at suitable ``delta``.
    """
    if h >= 0.5:
        return []
    rs = rho_star(P)
    out: List[Candidate] = []
    # family -> (user-indifference on high signal?, builder of (d_plus, d_minus)
    # from the user mixing weight w on the family's mixed signal).
    families = {
        "PI": (False, lambda w: (0.0, w)),  # sigma^U = (0, beta), mix low
        "HI": (True, lambda w: (w, 1.0)),  # sigma^U = (alpha, 1), mix high
    }
    for indiff_high, make_d in families.values():
        for sub in ("mix_high", "mix_low"):
            if sub == "mix_high":  # sigma^- = 1 corner, sigma^+ interior
                sp = _bisect(lambda s: _rho_tilde(h, mu, s, 1.0, P, high=indiff_high) - rs)
                if sp is None or not (CORNER < sp < 1 - CORNER):
                    continue
                sigma, rho_mix = (sp, 1.0), P.rho_plus
            else:  # sigma^+ = 0 corner, sigma^- interior
                sm = _bisect(lambda s: _rho_tilde(h, mu, 0.0, s, P, high=indiff_high) - rs)
                if sm is None or not (CORNER < sm < 1 - CORNER):
                    continue
                sigma, rho_mix = (0.0, sm), P.rho_minus
            u = reputation_updates(h, mu, sigma[0], sigma[1], P)

            def g(x, _u=u, _r=rho_mix, _mk=make_d):
                return aep_high_minus_low(_u, _r, *_mk(x), P)

            for w in _mixing_weights(g):
                d_plus, d_minus = make_d(w)
                out.append((d_plus, d_minus, sigma[0], sigma[1]))
    return out


def _mixing_weights(g: Callable[[float], float], flat_tol=1e-9) -> List[float]:
    """Interior user weights at which the agent is indifferent on its mixed signal.

    ``g`` is the agent's high-minus-low payoff as a function of the user's
    delegation probability on the mixed signal; it is affine. Generically there is
    a single interior root (the unique mixing weight). On a ``delta`` knife-edge
    (``kappa in {rho^-, rho^+}``) ``g`` is identically zero — the agent is
    indifferent for *every* weight, a continuum of equilibria — so we emit several
    interior representatives and let the best-response gate keep the valid ones.
    """
    g0, g1 = g(0.0), g(1.0)
    if np.isnan(g0) or np.isnan(g1):
        return []
    if abs(g0) < flat_tol and abs(g1) < flat_tol:
        return [0.25, 0.5, 0.75]
    r = _bisect(g)
    return [r] if r is not None and CORNER < r < 1 - CORNER else []


def generic_candidates(
    h: float, mu: float, P: Params, include_2mix=True, n_2mix=12, coarse_fallback=False
) -> List[Candidate]:
    """Candidate profiles from first principles (no transcribed rows).

    * Pure-user profiles: the 16 corners ``d, sigma in {0,1}^2`` (exact, grid-free)
      — standard, total-delegation, total-rejection, inverted.
    * One-mix boundary users: partial-standard and hedged-standard, each a pair of
      1-D root-finds (:func:`solve_partial_standard`, :func:`solve_hedged_standard`).
    * Two-mix boundary users (``include_2mix``): the inverted-favoring families
      (:func:`solve_inverted_boundary`).

    All candidates are validated downstream by the best-response gates, so an
    imperfect solve costs only coverage, never soundness.
    """
    cands: List[Candidate] = [
        (float(dp), float(dm), float(sp), float(sm))
        for dp, dm in itertools.product((0, 1), repeat=2)
        for sp, sm in itertools.product((0, 1), repeat=2)
    ]
    for solver in (solve_partial_standard, solve_hedged_standard):
        cand = solver(h, mu, P)
        if cand is not None:
            cands.append(cand)
    if include_2mix:
        cands.extend(solve_inverted_boundary(h, mu, P, n=n_2mix))
    if coarse_fallback:
        grid = np.linspace(0.0, 1.0, 11)
        cands.extend((dp, dm, sp, sm) for dp in (0.0, 1.0) for dm in grid for sp in grid for sm in (0.0, 1.0))
    return cands


def sampler_candidates(h: float, mu: float, P: Params, n=10) -> List[Candidate]:
    """Optional fast path: reuse the theory-pruned samplers from :mod:`numericals`
    purely as candidate generators (their own ``cond_*`` are ignored; our gates
    decide). Importing ``numericals`` is safe — its driver is under ``__main__``."""
    from . import numericals as nm

    cands: List[Candidate] = []
    for spec in nm.ROW_SPECS:
        if not spec["active"](P):
            continue
        try:
            users = spec["user_sampler"](spec["row"], n, P) or []
        except Exception:
            continue
        for dp, dm, _ in users:
            try:
                agents = spec["agent_sampler"](spec["row"], h, mu, dp, dm, n, P) or []
            except Exception:
                continue
            for sp, sm, _ in agents:
                cands.append((float(dp), float(dm), float(sp), float(sm)))
    return cands


# ============================================================
# Scan + cross-check + plot
# ============================================================


@dataclass
class ScanResult:
    P: Params
    by_label: Dict[str, List[Tuple[float, float]]]  # label -> [(h, mu), ...]
    n_grid: int


def equilibria_at(h, mu, P, source="generic", **kw) -> Dict[str, bool]:
    """Distinct descriptive labels of all equilibria at ``(h, mu)``."""
    if source == "samplers":
        cands = sampler_candidates(h, mu, P)
    else:
        cands = generic_candidates(h, mu, P, **kw)
    labels: Dict[str, bool] = {}
    for dp, dm, sp, sm in cands:
        if is_equilibrium(h, mu, dp, dm, sp, sm, P):
            labels[classify_equilibrium(dp, dm, sp, sm, P)] = True
    return labels


def scan(P: Params, n_grid=41, source="generic", **kw) -> ScanResult:
    hs = np.linspace(0.02, 0.98, n_grid)
    mus = np.linspace(0.02, 0.98, n_grid)
    by_label: Dict[str, List[Tuple[float, float]]] = {}
    for h in hs:
        for mu in mus:
            for label in equilibria_at(h, mu, P, source=source, **kw):
                by_label.setdefault(label, []).append((h, mu))
    return ScanResult(P=P, by_label=by_label, n_grid=n_grid)


def cross_check(P: Params, n_grid=25) -> Dict[str, int]:
    """Compare generic vs. sampler candidate sources cell-by-cell. Reports, per
    label, how many (h, mu) cells each source accepts and how many disagree. Zero
    disagreement validates the samplers' coverage against first principles."""
    hs = mus = np.linspace(0.02, 0.98, n_grid)
    counts = {"generic_only": 0, "samplers_only": 0, "both": 0}
    for h in hs:
        for mu in mus:
            g = set(equilibria_at(h, mu, P, source="generic", include_2mix=False))
            s = set(equilibria_at(h, mu, P, source="samplers"))
            counts["both"] += len(g & s)
            counts["generic_only"] += len(g - s)
            counts["samplers_only"] += len(s - g)
    return counts


def plot_equilibria(res: ScanResult, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    labels = sorted(res.by_label)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, label in enumerate(labels):
        pts = np.array(res.by_label[label])
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, color=cmap(i % 20), label=label)
    P = res.P
    h_grid = np.linspace(0.02, 0.98, 200)
    T = (rho_star(P) - P.rho_minus) / (P.rho_plus - rho_star(P))
    A_b = T * (1 - h_grid) / (1 + T * (1 - h_grid))
    mu_b = (A_b - P.theta_L) / (P.theta_H - P.theta_L)
    m = (mu_b >= 0) & (mu_b <= 1)
    ax.plot(h_grid[m], mu_b[m], "k-", lw=2, label=r"$\tau_C/\tau_D$ boundary")
    ax.set_xlabel(r"$h_1$")
    ax.set_ylabel(r"$\mu_1$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(rf"Equilibria ($\delta={P.delta}$, $\kappa={kappa(P):.3f}$)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path}")
    else:
        plt.show()


# ============================================================
# Demonstration: instances surfacing every realizable equilibrium type
# ============================================================

# Only hedged-standard is empty in the 2-type model at fixed (c, e, r): truthful
# low-reporting (sigma^-=0) makes a high report's delegation outcome uninformative,
# so V_success^+ = V_fail^+ and the fragile (c, 0) split it requires cannot arise.
# See the module docstring and revised_main.tex (prop:highuninform, cor:nohs).
IMPOSSIBLE_REGIMES = ("hedged-standard",)


def demo_instances() -> List[Tuple[str, Params]]:
    """Curated parameter instances (fixed c=0.2, e=0.5, r=1 -> rho^*=0.7) whose
    union surfaces every realizable equilibrium type."""
    mk = lambda **kw: Params(theta_L=0.1, theta_H=0.9, eps=1e-3, **kw)  # noqa: E731
    return [
        ("A: patient, robust tasks", mk(delta=0.05, rho_minus=0.2, rho_plus=0.8)),
        (
            "B: deflation onset (k=1-rho^+)",
            mk(delta=1 / 6, rho_minus=0.2, rho_plus=0.8),
        ),
        ("C: partial-standard band", mk(delta=0.4, rho_minus=0.4, rho_plus=0.95)),
        (
            "D: hedged-inverted (k>rho^+)",
            mk(delta=0.5, rho_minus=0.2, rho_plus=0.8),
        ),
    ]


def collect_examples(
    instances: List[Tuple[str, Params]], n_grid=61, tol=1e-5
) -> Dict[str, Tuple[str, float, float, Candidate]]:
    """One witness ``(instance, h, mu, profile)`` per descriptive equilibrium label.

    Two phases at a tight IC tolerance (so boundary artifacts of the loose region
    tolerance are excluded):
    (1) a coarse plane scan without the 2-mix solver — the open-region families
        (pure users + partial-standard);
    (2) a low-``h`` refine *with* the 2-mix solver — partial-inverted, which is a
        thin region requiring the both-signals-mixed agent.
    """
    found: Dict[str, Tuple[str, float, float, Candidate]] = {}

    def record(name, h, mu, P, **kw):
        for cand in generic_candidates(h, mu, P, **kw):
            if is_equilibrium(h, mu, *cand, P, tol=tol):
                label = classify_equilibrium(*cand, P)
                found.setdefault(
                    label,
                    (
                        name,
                        round(float(h), 3),
                        round(float(mu), 3),
                        tuple(round(x, 3) for x in cand),
                    ),
                )

    for name, P in instances:
        for h in np.linspace(0.02, 0.98, n_grid):
            for mu in np.linspace(0.02, 0.98, n_grid):
                record(name, h, mu, P, include_2mix=False)
        for h in np.linspace(0.02, 0.48, 24):  # inverted-favoring strip (h < 1/2)
            for mu in np.linspace(0.05, 0.97, 40):
                record(name, h, mu, P, include_2mix=True)
    return found


if __name__ == "__main__":
    instances = demo_instances()
    print("Parameters held fixed: c=0.2, e=0.5, r=1  ->  rho^* = 0.7\n")
    examples = collect_examples(instances)
    regimes = {lab.split(" · ")[0] for lab in examples}

    print(f"{'equilibrium type':38s} {'instance':12s} (h, mu)        profile (d+,d-,s+,s-)")
    print("-" * 92)
    for label in sorted(examples):
        name, h, mu, prof = examples[label]
        print(f"{label:38s} {name.split(':')[0]:12s} ({h:.3f}, {mu:.3f})   {prof}")

    print(f"\nRealizable user regimes ({len(regimes)}/8): {sorted(regimes)}")
    print(f"Empty / impossible (hedged-standard only): {list(IMPOSSIBLE_REGIMES)}")
