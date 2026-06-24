"""Reproducible figure: empirical over-/under-reporting vs delta with theory.

Regenerates the two-threshold figure (the ``mono_toy_rnd1`` plot) from a toy
experiment CSV produced by ``toy_experiment.run_experiments``, with the two
DERIVED thresholds drawn as vertical lines. The thresholds are computed a-priori
from game primitives (``rho_plus, rho_minus``) via ``toy_theory`` — they are NOT
fitted, which is the point of the overlay.

Usage
-----
    uv run python -m strategicmiscalibration.toy_plots \
        --data-dir outputs/experiments/sweep_toy_<ts> [--round 1] [--save]
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # batch figure generation, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .toy_model import ToyModelParams  # noqa: E402
from .toy_theory import (  # noqa: E402
    delta_deflation_onset,
    delta_inflation_onset,
    predicted_report_type,
)


def _latest_results_csv(data_dir: Path) -> Path:
    matches = sorted(glob.glob(str(data_dir / "results_*.csv")))
    if not matches:
        raise FileNotFoundError(f"No results_*.csv in {data_dir}")
    return Path(matches[-1])


def _params_from_df(df: pd.DataFrame) -> ToyModelParams:
    row = df.iloc[0]
    return ToyModelParams(
        rho_plus=float(row["rho_plus"]),
        rho_minus=float(row["rho_minus"]),
        theta_H=float(row["theta_H"]),
        theta_L=float(row["theta_L"]),
        reward=float(row["reward"]),
        cost=float(row["cost"]),
        effort=float(row["effort"]),
    )


def misreport_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-delta empirical over-reporting and sandbagging rates."""
    valid = df[(df["is_valid"]) & df["report_type"].notna()]
    g = valid.groupby("discount_factor")
    out = pd.DataFrame(
        {
            "n": g.size(),
            "overreport_rate": g["report_type"].apply(lambda s: (s == "OVERREPORTING").mean()),
            "sandbag_rate": g["report_type"].apply(lambda s: (s == "SANDBAGGING").mean()),
        }
    ).reset_index()
    return out.sort_values("discount_factor")


def theory_rates(df: pd.DataFrame, P: ToyModelParams) -> pd.DataFrame:
    """Theory-predicted over/under rates pushed through the SAME (state, delta)
    cells the empirical average uses, so the curves are comparable."""
    valid = df[(df["is_valid"]) & df["report_type"].notna()].copy()
    valid["pred"] = valid.apply(
        lambda r: predicted_report_type(bool(r["is_easy_task"]), float(r["discount_factor"]), P),
        axis=1,
    )
    g = valid.groupby("discount_factor")
    return (
        pd.DataFrame(
            {
                "pred_overreport_rate": g["pred"].apply(lambda s: (s == "OVERREPORTING").mean()),
                "pred_sandbag_rate": g["pred"].apply(lambda s: (s == "SANDBAGGING").mean()),
            }
        )
        .reset_index()
        .sort_values("discount_factor")
    )


def plot(data_dir: Path, round_filter: int | None = 1, save: bool = False) -> Path | None:
    csv = _latest_results_csv(data_dir)
    df = pd.read_csv(csv)
    if round_filter is not None and "round" in df.columns:
        df = df[df["round"] == round_filter]
    P = _params_from_df(df)

    emp = misreport_rates(df)
    th = theory_rates(df, P)
    d_def = delta_deflation_onset(P)
    d_inf = delta_inflation_onset(P)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.018
    ax.bar(
        emp["discount_factor"] - width / 2,
        emp["sandbag_rate"],
        width=width,
        color="tab:red",
        alpha=0.8,
        label="Sandbagging (empirical)",
    )
    ax.bar(
        emp["discount_factor"] + width / 2,
        emp["overreport_rate"],
        width=width,
        color="tab:blue",
        alpha=0.8,
        label="Over-reporting (empirical)",
    )

    ax.plot(
        th["discount_factor"],
        th["pred_sandbag_rate"],
        "--",
        color="darkred",
        marker="o",
        ms=3,
        lw=1,
        label="Sandbagging (theory)",
    )
    ax.plot(
        th["discount_factor"],
        th["pred_overreport_rate"],
        "--",
        color="navy",
        marker="o",
        ms=3,
        lw=1,
        label="Over-reporting (theory)",
    )

    ax.axvline(
        d_def,
        color="green",
        ls=":",
        lw=1.5,
        label=rf"$\delta$ at $\kappa=1-\rho^+$ ({d_def:.3f})",
    )
    ax.axvline(
        d_inf,
        color="purple",
        ls=":",
        lw=1.5,
        label=rf"$\delta$ at $\kappa=1-\rho^-$ ({d_inf:.3f})",
    )

    ax.set_xlabel(r"discount factor $\delta$ (large = myopic)")
    ax.set_ylabel("round-1 misreporting rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Two-threshold transition: derived thresholds (not fitted)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()

    if save:
        out = data_dir / "toy_two_threshold.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        return out
    plt.show()
    return None


# ============================================================
# Strategy map: sigma^A(rho) over the (h, mu) plane, 3 regimes vs theory
# (consumes outputs of strategicmiscalibration.toy_strategy_map)
# ============================================================

ACTION_VERSIONS = ("scaffolded", "minimal")
_VERSION_ORDER = ["scaffolded", "minimal", "strategy"]
_VERSION_TITLE = {
    "scaffolded": "scaffolded (original, action)",
    "minimal": "minimal (new, action)",
    "strategy": "strategy (direct elicitation)",
}


def _latest(data_dir: Path, stem: str) -> Path | None:
    m = sorted(glob.glob(str(data_dir / f"{stem}_*.csv")))
    return Path(m[-1]) if m else None


def _load_run_config(data_dir: Path) -> dict:
    m = sorted(glob.glob(str(data_dir / "config_*.json")))
    if not m:
        return {}
    with open(m[-1]) as f:
        return json.load(f)


def aggregate_strategy_map(df: pd.DataFrame) -> pd.DataFrame:
    """Per (version, h, mu, delta): empirical sigma_high, sigma_low and counts.

    Action versions: report frequency on EASY (-> sigma_high) and HARD (-> sigma_low).
    Strategy version: mean of the stated report-high probability for the observed rho_t,
    on EASY (-> sigma_high) and HARD (-> sigma_low) — same per-task structure."""
    if "is_valid" in df.columns:
        df = df[df["is_valid"].fillna(False)]
    out = []
    for (version, h, mu, delta), g in df.groupby(["version", "h", "mu", "delta"]):
        easy, hard = g[g.task == "EASY"], g[g.task == "HARD"]
        col = "report_high" if version in ACTION_VERSIONS else "strat_prob_high"
        sh, n_h = easy[col].mean(), int(easy[col].count())
        sl, n_l = hard[col].mean(), int(hard[col].count())
        out.append(
            dict(
                version=version,
                h=float(h),
                mu=float(mu),
                delta=float(delta),
                sigma_high=sh,
                sigma_low=sl,
                n_high=n_h,
                n_low=n_l,
            )
        )
    return pd.DataFrame(out)


def _axis_vals(agg: pd.DataFrame) -> tuple[list[float], list[float]]:
    return sorted(agg["h"].unique()), sorted(agg["mu"].unique())


def _pivot(agg: pd.DataFrame, version, delta, col, h_vals, mu_vals) -> np.ndarray:
    sub = agg[(agg.version == version) & np.isclose(agg.delta, delta)]
    M = np.full((len(mu_vals), len(h_vals)), np.nan)
    for _, r in sub.iterrows():
        M[mu_vals.index(r.mu), h_vals.index(r.h)] = r[col]
    return M


def _extent(h_vals, mu_vals):
    dh = (h_vals[1] - h_vals[0]) if len(h_vals) > 1 else 0.1
    dm = (mu_vals[1] - mu_vals[0]) if len(mu_vals) > 1 else 0.1
    return [h_vals[0] - dh / 2, h_vals[-1] + dh / 2, mu_vals[0] - dm / 2, mu_vals[-1] + dm / 2]


def _trust_overlay(ax, cfg: dict):
    """Draw the h=1/2 watershed and the Psi=Psi* trust boundary on a unit square."""
    ax.axvline(0.5, color="white", ls="--", lw=1.4, alpha=0.95)
    tf = cfg.get("toy_fixed", {})
    if not tf:
        return
    rs = cfg.get("rho_star", 1 - (tf["effort"] - tf["cost"]) / tf["reward"])
    psi_star = (rs - tf["rho_minus"]) / (tf["rho_plus"] - rs)
    hl = ml = np.linspace(1e-3, 0.999, 300)
    H, Mu = np.meshgrid(hl, ml)
    tbar = Mu * tf["theta_H"] + (1 - Mu) * tf["theta_L"]
    psi = (tbar / (1 - tbar)) / (1 - H)
    ax.contour(H, Mu, psi - psi_star, levels=[0.0], colors="black", linewidths=1.3)


def _heatmap(ax, M, cfg, title, cmap="viridis"):
    """One sigma heatmap over the full unit square (h, mu) in [0,1]^2; NaN -> gray."""
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.82")
    im = ax.imshow(
        np.ma.masked_invalid(M), origin="lower", extent=[0, 1, 0, 1], aspect="equal", vmin=0, vmax=1, cmap=cm
    )
    _trust_overlay(ax, cfg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("h (honesty belief)")
    ax.set_ylabel(r"$\mu$ (ability belief)")
    ax.set_title(title, fontsize=10)
    return im


def _theory_pivot(theory, delta, col, h_vals, mu_vals) -> np.ndarray:
    th = theory[np.isclose(theory.delta, delta)]
    M = np.full((len(mu_vals), len(h_vals)), np.nan)
    hi = {v: i for i, v in enumerate(h_vals)}
    mi = {v: i for i, v in enumerate(mu_vals)}
    for _, r in th.iterrows():
        if r.h in hi and r.mu in mi:
            M[mi[r.mu], hi[r.h]] = r[col]
    return M


def fig_unit_square(name, M_high, M_low, cfg, delta, out: Path):
    """Full unit-square (h, mu) heatmaps of the agent's report probability for each
    rho_1 realization: P(report HIGH | easy) and P(report HIGH | hard). One per setup
    (the three elicitation regimes and the theory prediction)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, Mv, lab in zip(
        axes,
        [M_high, M_low],
        [r"$\sigma^A(\rho^+)$ = P(report HIGH | easy task)", r"$\sigma^A(\rho^-)$ = P(report HIGH | hard task)"],
    ):
        im = _heatmap(ax, Mv, cfg, lab)
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(report HIGH)")
    fig.suptitle(
        f"{name}     |     "
        rf"$\delta$={delta:g}, $\kappa$={delta / (1 - delta):.2f}"
        "     (unit square; white dashed = watershed $h=\\frac{1}{2}$, "
        "black = trust boundary $\\Psi=\\Psi^*$; gray = no data)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_compare_versions(agg, cfg, delta, versions, h_vals, mu_vals, out: Path):
    """Grid: rows = {sigma_high, sigma_low}, cols = the three elicitation regimes."""
    nv = len(versions)
    fig, axes = plt.subplots(2, nv, figsize=(4.7 * nv, 9), squeeze=False)
    for j, version in enumerate(versions):
        for i, col in enumerate(["sigma_high", "sigma_low"]):
            im = _heatmap(
                axes[i][j],
                _pivot(agg, version, delta, col, h_vals, mu_vals),
                cfg,
                f"{version}: " + (r"$\sigma^A(\rho^+)$" if i == 0 else r"$\sigma^A(\rho^-)$"),
            )
            fig.colorbar(im, ax=axes[i][j], fraction=0.046)
    fig.suptitle(
        rf"Strategy across elicitation regimes  |  $\delta$={delta:g}"
        "   (white dashed = watershed $h=\\frac{1}{2}$, black = trust boundary)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_watershed_profiles(agg, theory, cfg, delta, versions, out: Path):
    """sigma^A(rho^+) vs h (averaged over mu), one line per version, with the h=1/2
    watershed and the (analytic) theory under-reporting region shaded. Headline
    test: does empirical under-reporting (sigma^+<1) stay confined to h < 1/2?"""
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    colors = {"scaffolded": "tab:blue", "minimal": "tab:green", "strategy": "tab:red"}
    # theory: largest h at which equilibria.py finds under-reporting (<= 1/2 by thm).
    th_h = None
    if theory is not None:
        u = theory[np.isclose(theory.delta, delta) & theory["pred_under_possible"]]
        th_h = float(u["h"].max()) if len(u) else None
    ax.axvspan(0.0, 0.5, color="orange", alpha=0.08, label="theory: under-reporting region (h<½)")
    if th_h is not None:
        ax.axvspan(0.0, th_h, color="red", alpha=0.08, label=f"theory: under found (h≤{th_h:.2f}, equilibria.py)")
    for version in versions:
        sub = agg[(agg.version == version) & np.isclose(agg.delta, delta)]
        g = sub.groupby("h")["sigma_high"].agg(["mean", "std", "count"]).reset_index()
        se = g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
        ax.errorbar(
            g["h"],
            g["mean"],
            yerr=se,
            marker="o",
            ms=4,
            capsize=3,
            color=colors.get(version),
            label=f"{version} (empirical)",
        )
    ax.axvline(0.5, color="black", ls="--", lw=1.3)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("h (honesty belief)")
    ax.set_ylabel(r"$\sigma^A(\rho^+)$ = P(report high | easy), averaged over $\mu$")
    ax.set_ylim(-0.02, 1.06)
    ax.set_title(
        rf"Trust watershed: under-reporting vs trust  |  $\delta$={delta:g}, "
        rf"$\kappa$={delta / (1 - delta):.2f}"
    )
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_theory_regions(theory, cfg, delta, out: Path):
    """Clean map of the equilibrium STRUCTURE over the unit square: the analytic
    trust regions (distrust / conditional-trust / blind-trust) as filled regions,
    the h=1/2 watershed, and the (thin, locus-borne) cells where equilibria.py finds
    payoff-relevant under-reporting. This is the theory side of the comparison."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    tf = cfg.get("toy_fixed", {})
    rs = cfg.get("rho_star", 1 - (tf["effort"] - tf["cost"]) / tf["reward"])
    psi_star = (rs - tf["rho_minus"]) / (tf["rho_plus"] - rs)
    hl = ml = np.linspace(1e-3, 0.999, 400)
    H, Mu = np.meshgrid(hl, ml)
    tbar = Mu * tf["theta_H"] + (1 - Mu) * tf["theta_L"]
    Omega = tbar / (1 - tbar)
    psi = Omega / (1 - H)
    blind = Omega * np.minimum(1.0, (1 - H) / np.maximum(H, 1e-9)) >= psi_star
    cat = np.where(blind, 2, np.where(psi >= psi_star, 1, 0))  # distrust/cond/blind

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = ListedColormap(["#f4e3c1", "#cfead0", "#a6cee3"])
    ax.imshow(cat, origin="lower", extent=[0, 1, 0, 1], aspect="equal", cmap=cmap, vmin=0, vmax=2)
    ax.axvline(0.5, color="black", ls="--", lw=1.5)
    ax.contour(H, Mu, psi - psi_star, levels=[0.0], colors="black", linewidths=1.0)

    u = theory[np.isclose(theory.delta, delta) & theory["pred_under_possible"]]
    handles = [
        Patch(facecolor="#f4e3c1", label=r"distrust $\tau_D$ ($V_2{=}0$)"),
        Patch(facecolor="#cfead0", label=r"conditional trust $\tau_C$"),
        Patch(facecolor="#a6cee3", label=r"blind trust $\tau_B$"),
    ]
    if len(u):
        ax.scatter(u["h"], u["mu"], s=22, marker="s", color="crimson", edgecolors="k", linewidths=0.3, zorder=5)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="crimson",
                markeredgecolor="k",
                markersize=8,
                label="under-reporting eq. (equilibria.py)",
            )
        )
    handles.append(Line2D([0], [0], color="black", ls="--", label=r"watershed $h=\frac{1}{2}$"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.92)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("h (honesty belief)")
    ax.set_ylabel(r"$\mu$ (ability belief)")
    ax.set_title(
        rf"Theory equilibrium structure  |  $\delta$={delta:g} "
        rf"($\kappa$={delta / (1 - delta):.2f})"
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Equilibrium feasible-region atlas + behavior-class comparison
# (built from the trusted equilibria.py theory grid, NOT the legacy
# numericals row-specs). These answer: where does each equilibrium live,
# and is the LLM's behavior on- or off-equilibrium at each (h, mu)?
# ------------------------------------------------------------------

# Distinct color per user-regime family for the atlas.
_FAMILY_COLOR = {
    "standard": "#1f77b4",
    "partial-standard": "#17becf",
    "inverted": "#8c564b",
    "partial-inverted": "#d62728",
    "hedged-inverted": "#e377c2",
    "hedged-standard": "#ff7f0e",
    "total-delegation": "#bcbd22",
    "total-rejection": "#7f7f7f",
    "interior-user": "#aec7e8",
}
# Families in which the agent's report is payoff-relevant (vs babbling).
_PAYOFF_RELEVANT = {
    "standard",
    "partial-standard",
    "inverted",
    "partial-inverted",
    "hedged-inverted",
    "hedged-standard",
}

# Agent-behavior panels for the comparison figure: (key, title, color, label-predicate).
# The predicate marks fine cells where SOME payoff-relevant equilibrium exhibits that
# behavior (parsed from equilibria.py's "regime · behavior" labels).
_BEHAVIOR_PANELS = [
    (
        "over",
        "over-reporting",
        "#2ca02c",
        lambda L: any(x.endswith("· over-reporting") and not x.startswith("total") for x in L),
    ),
    (
        "deflation",
        "clean deflation (under-report only)",
        "#d62728",
        lambda L: any(x.endswith("· under-reporting") and not x.startswith("total") for x in L),
    ),
    (
        "inverted",
        "inverted (over + under)",
        "#9467bd",
        lambda L: any(x.endswith("· over+under") and not x.startswith("total") for x in L),
    ),
    (
        "honest",
        "honest separation",
        "#1f77b4",
        lambda L: any(x.endswith("· honest") and not x.startswith("total") for x in L),
    ),
]
_LLM_MARKER = {"scaffolded": "o", "minimal": "s", "strategy": "^"}
_LLM_JITTER = {"scaffolded": -0.022, "minimal": 0.0, "strategy": 0.022}


def _theory_axes(theory) -> tuple[np.ndarray, np.ndarray]:
    return np.array(sorted(theory["h"].unique())), np.array(sorted(theory["mu"].unique()))


def _theory_mask(theory, delta, column, predicate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean (n_mu, n_h) grid: predicate(set of pipe-split tokens in ``column``)."""
    H, M = _theory_axes(theory)
    sub = theory[np.isclose(theory.delta, delta)]
    Z = np.zeros((len(M), len(H)), bool)
    for _, r in sub.iterrows():
        toks = set(str(r[column]).split("|")) if isinstance(r[column], str) else set()
        Z[int(np.argmin(np.abs(M - r.mu))), int(np.argmin(np.abs(H - r.h)))] = predicate(toks)
    return H, M, Z


def _trust_boundary_xy(cfg: dict):
    tf = cfg.get("toy_fixed", {})
    rs = cfg.get("rho_star", 1 - (tf["effort"] - tf["cost"]) / tf["reward"])
    T = (rs - tf["rho_minus"]) / (tf["rho_plus"] - rs)
    hg = np.linspace(0, 1, 200)
    A = T * (1 - hg) / (1 + T * (1 - hg))
    mu = (A - tf["theta_L"]) / (tf["theta_H"] - tf["theta_L"])
    ok = (mu >= 0) & (mu <= 1)
    return hg[ok], mu[ok]


def _region_axes(ax, cfg: dict):
    """Common framing for region plots: trust boundary (black) + h=1/2 (dark dashed)."""
    hb, mb = _trust_boundary_xy(cfg)
    ax.plot(hb, mb, "k-", lw=2, zorder=5)
    ax.axvline(0.5, color="0.35", ls="--", lw=1.4, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("h (honesty belief)")


def _llm_behavior_class(sh: float, sl: float) -> str | None:
    """Coarse agent-behavior class from (sigma_high, sigma_low). Thresholds 0.9/0.1
    are deliberately loose given small per-cell samples."""
    if sh is None or sl is None or np.isnan(sh) or np.isnan(sl):
        return None
    if sh >= 0.9 and sl <= 0.1:
        return "honest"
    if sh >= 0.9 and sl > 0.1:
        return "over"
    if sh < 0.9 and sl <= 0.1:
        return "deflation"
    return "inverted"


def fig_equilibrium_atlas(theory, cfg, delta, out: Path):
    """Notebook-style atlas: one panel per equilibrium family, with that family's
    feasible (h, mu) region shaded (from equilibria.py), plus the trust boundary and
    watershed. Makes visible that clean deflation and honesty are feasible nowhere."""
    fams = [f for f in _FAMILY_COLOR if _theory_mask(theory, delta, "regimes", lambda S, f=f: f in S)[2].any()]
    if not fams:
        return
    n = len(fams)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    for ax, f in zip(axes[0], fams):
        H, M, Z = _theory_mask(theory, delta, "regimes", lambda S, f=f: f in S)
        ax.contourf(H, M, Z.astype(float), levels=[0.5, 1.5], colors=[_FAMILY_COLOR[f]], alpha=0.55, zorder=2)
        _region_axes(ax, cfg)
        kind = "payoff-relevant" if f in _PAYOFF_RELEVANT else "babbling"
        ax.set_title(f"{f}\n({kind})", fontsize=9)
    axes[0][0].set_ylabel(r"$\mu$ (ability belief)")
    fig.suptitle(
        rf"Equilibrium feasible regions (equilibria.py)  |  $\delta$={delta:g} "
        rf"($\kappa$={delta / (1 - delta):.2f}).  black=$\Psi=\Psi^*$, dashed=$h=1/2$.  "
        "Clean deflation & honesty are feasible nowhere.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_behavior_feasibility_vs_llm(agg, theory, cfg, delta, versions, out: Path):
    """Per agent-behavior class (over / clean-deflation / inverted / honest): shade
    where it is a payoff-relevant equilibrium (color) and where it is babbling-only
    (light grey), then overlay the LLM cells exhibiting that class (one marker per
    setup). A marker in an unshaded area = off-equilibrium behavior. This is the
    direct equilibria-vs-LLM comparison."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    babble = lambda S: any(x.startswith(("total-rejection", "total-delegation")) for x in S)  # noqa: E731
    fig, axes = plt.subplots(1, len(_BEHAVIOR_PANELS), figsize=(15, 3.9), squeeze=False)
    for ax, (cls, title, col, pred) in zip(axes[0], _BEHAVIOR_PANELS):
        H, M, Zb = _theory_mask(theory, delta, "labels", babble)
        ax.contourf(H, M, Zb.astype(float), levels=[0.5, 1.5], colors=["0.8"], alpha=0.5, zorder=1)
        H, M, Zp = _theory_mask(theory, delta, "labels", pred)
        if Zp.any():
            ax.contourf(H, M, Zp.astype(float), levels=[0.5, 1.5], colors=[col], alpha=0.55, zorder=2)
        _region_axes(ax, cfg)
        for v in versions:
            sub = agg[(agg.version == v) & np.isclose(agg.delta, delta)]
            xs = [
                r.h + _LLM_JITTER.get(v, 0.0)
                for _, r in sub.iterrows()
                if _llm_behavior_class(r.sigma_high, r.sigma_low) == cls
            ]
            ys = [r.mu for _, r in sub.iterrows() if _llm_behavior_class(r.sigma_high, r.sigma_low) == cls]
            ax.scatter(
                xs,
                ys,
                marker=_LLM_MARKER.get(v, "o"),
                s=55,
                facecolor="none",
                edgecolor="black",
                linewidths=1.5,
                zorder=6,
            )
        ax.set_title(title, fontsize=9.5)
    axes[0][0].set_ylabel(r"$\mu$ (ability belief)")
    leg = [
        Patch(facecolor="0.8", alpha=0.5, label="babbling region (any σ is eq.)"),
        Patch(facecolor="grey", alpha=0.55, label="behavior is payoff-relevant eq. (shaded)"),
        Line2D([0], [0], color="k", lw=2, label=r"$\Psi=\Psi^*$"),
        Line2D([0], [0], color="0.35", ls="--", label="$h=1/2$"),
    ]
    leg += [
        Line2D([0], [0], marker=_LLM_MARKER[v], ls="", mfc="none", mec="k", mew=1.5, ms=8, label=f"LLM: {v}")
        for v in versions
        if v in _LLM_MARKER
    ]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        rf"Which agent behaviors are equilibria, and where the LLM exhibits them  |  "
        rf"$\delta$={delta:g} ($\kappa$={delta / (1 - delta):.2f}).  "
        "A marker in an UNSHADED area is off-equilibrium behavior.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_inflation_delta_contrast(agg, cfg, deltas, versions, out: Path):
    """sigma_low (over-reporting on HARD) vs h, patient vs myopic delta overlaid, one
    panel per setup. Visualizes the inflation force: over-reporting should intensify
    toward myopia (kappa -> 1-rho^-). The patience axis the (h,mu) maps cannot show."""
    tf = cfg.get("toy_fixed", {})
    infl = 1 - tf["rho_minus"] if tf else None
    lo, hi = deltas[0], deltas[-1]
    colors = {lo: "tab:blue", hi: "tab:red"}
    fig, axes = plt.subplots(1, len(versions), figsize=(4.2 * len(versions), 4.0), squeeze=False, sharey=True)
    for ax, v in zip(axes[0], versions):
        for d in (lo, hi):
            sub = agg[(agg.version == v) & np.isclose(agg.delta, d)]
            g = sub.groupby("h")["sigma_low"].agg(["mean", "std", "count"]).reset_index()
            se = g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
            ax.errorbar(
                g["h"],
                g["mean"],
                yerr=se,
                marker="o",
                capsize=3,
                color=colors.get(d, None),
                label=rf"$\delta$={d:g} ($\kappa$={d / (1 - d):.2f})",
            )
        ax.axvline(0.5, color="grey", ls=":", lw=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("h (honesty belief)")
        ax.set_title(v, fontsize=10)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel(r"$\sigma^A(\rho^-)$ = P(report HIGH | HARD)")
    tail = rf"; theory inflation onset $\kappa$=1$-\rho^-$={infl:.2f}" if infl is not None else ""
    fig.suptitle(
        r"Inflation force: over-reporting on HARD draws rises toward myopia"
        rf" ($\kappa$: {lo / (1 - lo):.2f}$\to${hi / (1 - hi):.2f}{tail})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def make_strategy_figures(data_dir: Path, save: bool = True) -> Path:
    """Generate the full set of strategy-map figures into ``data_dir/figures/``."""
    data_dir = Path(data_dir)
    emp_csv = _latest(data_dir, "strategy_map")
    if emp_csv is None:
        raise FileNotFoundError(f"No strategy_map_*.csv in {data_dir}")
    th_csv = _latest(data_dir, "theory_grid")
    cfg = _load_run_config(data_dir)
    agg = aggregate_strategy_map(pd.read_csv(emp_csv))
    theory = pd.read_csv(th_csv) if th_csv else None
    h_vals, mu_vals = _axis_vals(agg)
    versions = [v for v in _VERSION_ORDER if v in agg["version"].unique().tolist()]
    deltas = sorted(agg["delta"].unique())

    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def D(d):  # delta tag for filenames
        return f"delta{d:g}".replace(".", "p")

    th_h = sorted(theory["h"].unique()) if theory is not None else []
    th_mu = sorted(theory["mu"].unique()) if theory is not None else []

    for delta in deltas:
        # Unit-square reporting heatmap, one per setup (the 3 regimes + theory).
        for version in versions:
            p = fig_dir / f"unit_square_{version}_{D(delta)}.pdf"
            fig_unit_square(
                _VERSION_TITLE.get(version, version),
                _pivot(agg, version, delta, "sigma_high", h_vals, mu_vals),
                _pivot(agg, version, delta, "sigma_low", h_vals, mu_vals),
                cfg,
                delta,
                p,
            )
            written.append(p.name)
        if theory is not None:
            p = fig_dir / f"unit_square_theory_{D(delta)}.pdf"
            fig_unit_square(
                "theory (equilibria.py): most-under σ⁺ / most-over σ⁻ across equilibria",
                _theory_pivot(theory, delta, "pred_sigma_high_lo", th_h, th_mu),
                _theory_pivot(theory, delta, "pred_sigma_low_hi", th_h, th_mu),
                cfg,
                delta,
                p,
            )
            written.append(p.name)

        p = fig_dir / f"compare_versions_{D(delta)}.pdf"
        fig_compare_versions(agg, cfg, delta, versions, h_vals, mu_vals, p)
        written.append(p.name)
        p = fig_dir / f"watershed_profiles_{D(delta)}.pdf"
        fig_watershed_profiles(agg, theory, cfg, delta, versions, p)
        written.append(p.name)
        if theory is not None:
            p = fig_dir / f"theory_regions_{D(delta)}.pdf"
            fig_theory_regions(theory, cfg, delta, p)
            written.append(p.name)
            p = fig_dir / f"equilibrium_atlas_{D(delta)}.pdf"
            fig_equilibrium_atlas(theory, cfg, delta, p)
            written.append(p.name)
            p = fig_dir / f"behavior_feasibility_vs_llm_{D(delta)}.pdf"
            fig_behavior_feasibility_vs_llm(agg, theory, cfg, delta, versions, p)
            written.append(p.name)

    # Patience-axis contrast (needs >= 2 deltas): over-reporting vs myopia.
    if len(deltas) >= 2:
        p = fig_dir / "inflation_delta_contrast.pdf"
        fig_inflation_delta_contrast(agg, cfg, deltas, versions, p)
        written.append(p.name)

    agg.to_csv(fig_dir / "aggregated_strategy.csv", index=False)
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": emp_csv.name,
        "theory_csv": th_csv.name if th_csv else None,
        "versions": versions,
        "deltas": deltas,
        "h_grid": h_vals,
        "mu_grid": mu_vals,
        "run_config": cfg,
        "figures": {
            "unit_square_<setup>_<delta>": "full unit-square (h,μ) heatmaps of P(report HIGH|easy) and P(report HIGH|hard), one per setup (scaffolded/minimal/strategy/theory)",
            "compare_versions_<delta>": "the three elicitation regimes side by side (σ⁺ and σ⁻), watershed + trust-boundary overlays",
            "watershed_profiles_<delta>": "HEADLINE: σ⁺ vs h (avg over μ) per regime, with h=½ watershed + theory under-region shaded",
            "theory_regions_<delta>": "equilibrium structure (equilibria.py): trust regions τ_D/τ_C/τ_B filled, watershed, and under-reporting equilibria marked",
            "equilibrium_atlas_<delta>": "one panel per equilibrium family with its feasible (h,μ) region color-coded (equilibria.py); clean deflation & honesty feasible nowhere",
            "behavior_feasibility_vs_llm_<delta>": "per agent-behavior class: where it is a payoff-relevant eq. (color) vs babbling-only (grey), with LLM cells overlaid; marker in unshaded area = off-equilibrium",
            "inflation_delta_contrast": "σ⁻=P(HIGH|HARD) vs h, patient vs myopic δ overlaid per setup; the inflation/patience axis (over-reporting rises toward myopia)",
        },
        "files": written,
    }
    with open(fig_dir / "figures_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {len(written)} figures + manifest to {fig_dir}")
    return fig_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--save", action="store_true")
    ap.add_argument(
        "--kind",
        choices=["auto", "two-threshold", "strategy"],
        default="auto",
        help="auto: strategy-map if strategy_map_*.csv present, else two-threshold",
    )
    args = ap.parse_args()

    kind = args.kind
    if kind == "auto":
        kind = "strategy" if _latest(args.data_dir, "strategy_map") else "two-threshold"
    if kind == "strategy":
        make_strategy_figures(args.data_dir, save=True)
    else:
        plot(args.data_dir, round_filter=args.round, save=args.save)


if __name__ == "__main__":
    main()
