"""Map the agent's first-round reporting strategy sigma^A(rho) over the (h, mu)
prior plane, under three elicitation regimes.

Motivation
----------
The revised theory predicts behaviour as a function of the reputation state
(h, mu): in the binary model, no payoff-relevant high-side under-reporting for
h >= 1/2 (the *trust watershed*, ``thm:phase``(c)), with deflation confined to the
low-trust region h < 1/2. This runner elicits ``sigma^A(rho^+)`` and
``sigma^A(rho^-)`` cell-by-cell so the watershed (and any trusted-region
sandbagging, which would support the imported informative-failure prior) can be
read off directly.

Three regimes, all in the toy setting, all eliciting the SAME object
(``sigma_high = P(report high | easy)``, ``sigma_low = P(report high | hard)``):

* ``scaffolded`` — original toy agent prompt, ACTION elicitation. At each cell we
  draw an EASY task (to measure ``sigma_high``) and a HARD task (``sigma_low``)
  and sample the binary report ``n_action`` times at ``temperature>0``; the report
  frequency estimates the mixing probability.
* ``minimal`` — the de-scaffolded ("new") toy agent prompt, same ACTION protocol.
  The scaffolded/minimal contrast isolates prompt-leakage artifacts.
* ``strategy`` — same information structure as the action regimes (the agent
  OBSERVES its current ``rho_t``, matching the game), but instead of a sampled binary
  report it states the *probability* with which it would report HIGH for that observed
  ``rho_t``. Called per task (EASY -> ``sigma_high``, HARD -> ``sigma_low``), sampled
  ``n_strategy`` times to average the stated probability and its spread. The
  action-vs-strategy contrast is then a clean stated-vs-revealed check at the *same*
  decision node (``notes/strategy_elicitation_scope.md``).

Each elicitation is a single round-1 (non-final) call: round 1 of a 2-round game
so the agent faces a genuine reputational future, but we never play round 2.

Output: a tidy per-call CSV under ``outputs/experiments/strategy_map_<ts>/``.
Use ``--dry-run`` to print the call budget and runtime estimate without calling.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .datatypes import TEMPLATE_DIR
from .llm_interface import (
    OutputMode,
    ToyAgentSignalResponse,
    ToyAgentStrategyResponse,
    answer_only_schema,
    configure_logging,
    load_template,
    query_llm,
    reasoning_native,
)
from .toy_experiment import DEFAULT_MODEL, ToyExperimentConfig

logger = logging.getLogger(__name__)

STRATEGY_TEMPLATE = TEMPLATE_DIR / "toy/strategy_agent_prompt.j2"

# Central per-call latency for the runtime estimate. MEASURED ~10s/call for
# Together Llama-3.3-70B-Turbo with reasoning + structured output, sequential.
SEC_PER_CALL = 10.0

# Defaults sized for ~4h at SEC_PER_CALL (see _budget()): 5x5 grid x 2 delta x
# {scaffolded:6, minimal:6, strategy:5} = 1450 calls ~= 4.0h @ 10s/call.
DEFAULT_H_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_MU_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_DELTAS = [0.15, 0.55]  # patient (sandbagging regime) and myopic (inflation)
DEFAULT_N_ACTION = 5
DEFAULT_N_STRATEGY = 1
ACTION_VERSIONS = ("scaffolded", "minimal")
ALL_VERSIONS = ("scaffolded", "minimal", "strategy")
THEORY_RES = 31  # fine (h, mu) resolution for the equilibria.py theory grid / overlays

# The toy environment's fixed primitives (rho* = 1 - (e-c)/r = 0.6). Shared by the
# elicitation configs and the equilibria.py theory grid so the two are comparable.
TOY_FIXED = dict(
    rho_plus=0.85,
    rho_minus=0.15,
    theta_H=0.8,
    theta_L=0.2,
    reward=1.0,
    cost=0.1,
    effort=0.5,
)


def _mk_cfg(
    h: float,
    mu: float,
    delta: float,
    prompt_style: str,
    temperature: float,
    model: str,
    output_mode: OutputMode = OutputMode.JSON_SCHEMA,
    max_tokens: int = 512,
) -> ToyExperimentConfig:
    """A param bundle for one cell (reuses the toy fixed parameters)."""
    return ToyExperimentConfig(
        user_model_name=model,
        agent_model_name=model,
        user_kind="oracle",
        agent_prompt_style=prompt_style if prompt_style in ACTION_VERSIONS else "minimal",
        num_trials=1,
        num_rounds=2,
        temperature=temperature,
        discount_factor=delta,
        h_0=h,
        mu_0=mu,
        first_round_task="EASY",
        agent_eta=0,
        agent_theta_kind="H",
        seed=12345,
        output_mode=output_mode,
        max_tokens=max_tokens,
        **TOY_FIXED,
    )


# Regimes in which the agent's report affects its payoff (so sigma^+<1 is genuine
# high-side under-reporting, not babbling under blind trust / total rejection).
_PAYOFF_REGIMES = frozenset(
    {
        "standard",
        "partial-standard",
        "inverted",
        "partial-inverted",
        "hedged-inverted",
        "hedged-standard",
    }
)


def _toy_params(delta: float):
    """numericals.Params at the toy primitives (consumed by equilibria.py)."""
    from .numericals import Params

    return Params(
        theta_L=TOY_FIXED["theta_L"],
        theta_H=TOY_FIXED["theta_H"],
        rho_minus=TOY_FIXED["rho_minus"],
        rho_plus=TOY_FIXED["rho_plus"],
        r=TOY_FIXED["reward"],
        e=TOY_FIXED["effort"],
        c=TOY_FIXED["cost"],
        delta=delta,
        eps=1e-9,
    )


def theory_strategy_grid(h_grid: List[float], mu_grid: List[float], deltas: List[float]) -> pd.DataFrame:
    """Binary-equilibrium predicted agent strategy over the (h, mu, delta) grid,
    computed from first principles via :mod:`equilibria`. For each cell we scan all
    PBE profiles and summarize the predicted sigma^A(rho^+), sigma^A(rho^-): the
    range across equilibria, and whether payoff-relevant under-/over-reporting is
    possible. This is the theory side the empirical map is compared against."""
    from . import equilibria as eq

    corner = eq.CORNER
    rows: List[Dict[str, Any]] = []
    for delta in deltas:
        P = _toy_params(delta)
        for h in h_grid:
            for mu in mu_grid:
                # Union generic (root-find) + sampler (closed-form boundary) candidates
                # for coverage, then keep those passing the BR gates (sound regardless).
                raw = list(eq.generic_candidates(h, mu, P, include_2mix=True))
                raw += list(eq.sampler_candidates(h, mu, P))
                seen, cands = set(), []
                for c in raw:
                    key = tuple(round(x, 6) for x in c)
                    if key in seen or not eq.is_equilibrium(h, mu, *c, P, tol=1e-3):
                        continue
                    seen.add(key)
                    cands.append(c)
                regimes = sorted({eq._user_regime(c[0], c[1]) for c in cands})
                labels = sorted({eq.classify_equilibrium(*c, P) for c in cands})
                pr = [c for c in cands if eq._user_regime(c[0], c[1]) in _PAYOFF_REGIMES]
                sh = [c[2] for c in pr]  # sigma^+ (report-high prob on easy)
                sl = [c[3] for c in pr]  # sigma^- (report-high prob on hard)
                rows.append(
                    {
                        "h": h,
                        "mu": mu,
                        "delta": delta,
                        "kappa": eq.kappa(P),
                        "n_eq": len(cands),
                        "regimes": "|".join(regimes),
                        "labels": "|".join(labels),
                        "pred_under_possible": any(s < 1 - corner for s in sh),
                        "pred_over_possible": any(s > corner for s in sl),
                        "pred_sigma_high_lo": min(sh) if sh else float("nan"),
                        "pred_sigma_high_hi": max(sh) if sh else float("nan"),
                        "pred_sigma_low_lo": min(sl) if sl else float("nan"),
                        "pred_sigma_low_hi": max(sl) if sl else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _query_agent(cfg: ToyExperimentConfig, template, schema, **tmpl_kwargs):
    """Query the agent, adapting to reasoning-native models. For those (e.g. gpt-oss
    in TEXT mode) we request only the answer field(s) and read the chain-of-thought
    from the API ``reasoning_content`` channel, so a long in-body ``reasoning`` field
    cannot truncate the trailing answer. Returns (parsed_answer, reasoning_text)."""
    native = reasoning_native(cfg.agent_model_name) and cfg.output_mode == OutputMode.TEXT
    parse_schema = answer_only_schema(schema) if native else schema
    prompt = load_template(
        template,
        output_mode=cfg.output_mode,
        response_schema=parse_schema,
        **tmpl_kwargs,
    )
    if native:
        ans, _, reasoning = query_llm(
            cfg.agent_model_name,
            prompt,
            parse_schema,
            cfg.max_tokens,
            cfg.temperature,
            output_mode=cfg.output_mode,
            return_reasoning=True,
        )
        return ans, reasoning
    resp, _ = query_llm(
        cfg.agent_model_name,
        prompt,
        parse_schema,
        cfg.max_tokens,
        cfg.temperature,
        output_mode=cfg.output_mode,
    )
    return resp, resp.reasoning


def _elicit_action(cfg: ToyExperimentConfig, h: float, mu: float, rho_t: float) -> Tuple[Optional[bool], float, str]:
    """One round-1 ACTION call. Returns (report_high|None, signal, reasoning)."""
    resp, reasoning = _query_agent(
        cfg,
        cfg.agent_template_for(1),  # round 1 of 2 -> non-final, reputational future
        ToyAgentSignalResponse,
        round_num=1,
        total_rounds=cfg.num_rounds,
        discount_factor=cfg.discount_factor,
        reward=cfg.reward,
        cost=cfg.cost,
        effort=cfg.effort,
        rho_plus=cfg.rho_plus,
        rho_minus=cfg.rho_minus,
        theta_H=cfg.theta_H,
        theta_L=cfg.theta_L,
        rho_t=rho_t,
        h_0=h,
        mu_0=mu,
        agent_type_desc=cfg.agent_type_desc,
    )
    if abs(resp.signal - cfg.rho_plus) < 1e-9:
        return True, resp.signal, reasoning
    if abs(resp.signal - cfg.rho_minus) < 1e-9:
        return False, resp.signal, reasoning
    return None, resp.signal, reasoning


def _elicit_strategy(cfg: ToyExperimentConfig, h: float, mu: float, rho_t: float) -> Tuple[float, str]:
    """One STRATEGY call for the observed ``rho_t``. Returns (prob_high, reasoning),
    clipped to [0,1]. The agent observes ``rho_t`` (as in the game) and states its
    probability of reporting HIGH for it."""
    resp, reasoning = _query_agent(
        cfg,
        STRATEGY_TEMPLATE,
        ToyAgentStrategyResponse,
        round_num=1,
        total_rounds=cfg.num_rounds,
        discount_factor=cfg.discount_factor,
        reward=cfg.reward,
        cost=cfg.cost,
        effort=cfg.effort,
        rho_plus=cfg.rho_plus,
        rho_minus=cfg.rho_minus,
        theta_H=cfg.theta_H,
        theta_L=cfg.theta_L,
        rho_t=rho_t,
        h_0=h,
        mu_0=mu,
        agent_type_desc=cfg.agent_type_desc,
    )
    clip = lambda x: max(0.0, min(1.0, float(x)))  # noqa: E731
    return clip(resp.prob_high), reasoning


def _budget(versions, h_grid, mu_grid, deltas, n_action, n_strategy) -> Dict[str, int]:
    cells = len(h_grid) * len(mu_grid) * len(deltas)
    per: Dict[str, int] = {}
    for v in versions:
        # both action and strategy regimes now query each cell per task (EASY, HARD)
        per[v] = cells * 2 * (n_action if v in ACTION_VERSIONS else n_strategy)
    per["TOTAL"] = sum(per[v] for v in versions)
    return per


def run_map(
    versions: List[str],
    h_grid: List[float],
    mu_grid: List[float],
    deltas: List[float],
    n_action: int,
    n_strategy: int,
    temperature: float,
    model: str,
    out_dir: Path,
    output_mode: OutputMode = OutputMode.JSON_SCHEMA,
    max_tokens: int = 512,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    budget = _budget(versions, h_grid, mu_grid, deltas, n_action, n_strategy)
    bar = tqdm(total=budget["TOTAL"], desc="strategy map")

    def base(version, h, mu, delta) -> Dict[str, Any]:
        return {
            "version": version,
            "h": h,
            "mu": mu,
            "delta": delta,
            "kappa": delta / (1.0 - delta),
            "temperature": temperature,
            "model": model,
        }

    for version in versions:
        for delta in deltas:
            for h in h_grid:
                for mu in mu_grid:
                    cfg = _mk_cfg(
                        h,
                        mu,
                        delta,
                        version,
                        temperature,
                        model,
                        output_mode=output_mode,
                        max_tokens=max_tokens,
                    )
                    if version in ACTION_VERSIONS:
                        for task, rho_t in (
                            ("EASY", cfg.rho_plus),
                            ("HARD", cfg.rho_minus),
                        ):
                            for i in range(n_action):
                                row = {
                                    **base(version, h, mu, delta),
                                    "task": task,
                                    "rho_t": rho_t,
                                    "sample_idx": i,
                                }
                                try:
                                    rh, sig, why = _elicit_action(cfg, h, mu, rho_t)
                                    row.update(
                                        signal=sig,
                                        report_high=(None if rh is None else int(rh)),
                                        reasoning=why,
                                        is_valid=rh is not None,
                                    )
                                except Exception as e:  # noqa: BLE001
                                    logger.warning(
                                        "action call failed (%s h=%s mu=%s d=%s %s): %s",
                                        version,
                                        h,
                                        mu,
                                        delta,
                                        task,
                                        e,
                                    )
                                    row.update(is_valid=False)
                                rows.append(row)
                                bar.update(1)
                    else:  # strategy: stated report-high probability for the observed rho_t
                        for task, rho_t in (
                            ("EASY", cfg.rho_plus),
                            ("HARD", cfg.rho_minus),
                        ):
                            for i in range(n_strategy):
                                row = {
                                    **base(version, h, mu, delta),
                                    "task": task,
                                    "rho_t": rho_t,
                                    "sample_idx": i,
                                }
                                try:
                                    ph, why = _elicit_strategy(cfg, h, mu, rho_t)
                                    row.update(
                                        strat_prob_high=ph,
                                        reasoning=why,
                                        is_valid=True,
                                    )
                                except Exception as e:  # noqa: BLE001
                                    logger.warning(
                                        "strategy call failed (h=%s mu=%s d=%s %s): %s",
                                        h,
                                        mu,
                                        delta,
                                        task,
                                        e,
                                    )
                                    row.update(is_valid=False)
                                rows.append(row)
                                bar.update(1)
    bar.close()

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"strategy_map_{ts}.csv", index=False)

    # Theory side: binary-equilibrium predicted sigma^A on a FINE grid (equilibria.py),
    # decoupled from the coarse empirical grid so boundary equilibria (which live on
    # loci) are captured for the overlays.
    print("\nComputing theory grid (equilibria.py) for comparison ...")
    fine = [round(x, 3) for x in (0.05 + 0.9 * i / (THEORY_RES - 1) for i in range(THEORY_RES))]
    theory = theory_strategy_grid(fine, fine, deltas)
    theory.to_csv(out_dir / f"theory_grid_{ts}.csv", index=False)

    with open(out_dir / f"config_{ts}.json", "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "kind": "strategy_map",
                "versions": versions,
                "h_grid": h_grid,
                "mu_grid": mu_grid,
                "deltas": deltas,
                "n_action": n_action,
                "n_strategy": n_strategy,
                "temperature": temperature,
                "model": model,
                "output_mode": output_mode.value,
                "max_tokens": max_tokens,
                "toy_fixed": TOY_FIXED,
                "rho_star": 1 - (TOY_FIXED["effort"] - TOY_FIXED["cost"]) / TOY_FIXED["reward"],
                "budget": budget,
                "n_rows": len(df),
            },
            f,
            indent=2,
            default=str,
        )
    _print_summary(df)

    # Figures: empirical maps per version + cross-version + theory comparison.
    try:
        from . import toy_plots

        toy_plots.make_strategy_figures(out_dir, save=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Figure generation failed (data is saved): %s", e)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Per-version sigma_high / sigma_low, marginalized over mu, by (h, delta)."""
    if df.empty:
        return
    print("\n=== sigma_high (P report high | EASY) by version, h, delta ===")
    for version in sorted(df["version"].unique()):
        sub = df[(df.version == version) & df.is_valid.fillna(False)]
        easy = sub[sub.task == "EASY"]
        if version in ACTION_VERSIONS:
            g = easy.groupby(["delta", "h"]).report_high.mean()
        else:
            g = easy.groupby(["delta", "h"]).strat_prob_high.mean()
        print(f"\n[{version}]")
        print(g.round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Map sigma^A(rho) over the (h, mu) plane.")
    ap.add_argument(
        "--versions",
        default=",".join(ALL_VERSIONS),
        help="comma list of {scaffolded,minimal,strategy}",
    )
    ap.add_argument("--h-grid", default=",".join(map(str, DEFAULT_H_GRID)))
    ap.add_argument("--mu-grid", default=",".join(map(str, DEFAULT_MU_GRID)))
    ap.add_argument("--deltas", default=",".join(map(str, DEFAULT_DELTAS)))
    ap.add_argument("--n-action", type=int, default=DEFAULT_N_ACTION)
    ap.add_argument("--n-strategy", type=int, default=DEFAULT_N_STRATEGY)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--output-mode",
        choices=["auto", "json_schema", "text"],
        default="auto",
        help="how structured output is obtained; 'auto' uses TEXT for models that "
        "handle native JSON-schema poorly (e.g. gpt-oss), else JSON_SCHEMA",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="per-call token cap (default 512 for json_schema, 1024 for text to "
        "leave room for reasoning + JSON and avoid truncated/EOF responses)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the call budget and runtime estimate, make no calls",
    )
    args = ap.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    h_grid = [float(x) for x in args.h_grid.split(",")]
    mu_grid = [float(x) for x in args.mu_grid.split(",")]
    deltas = [float(x) for x in args.deltas.split(",")]
    budget = _budget(versions, h_grid, mu_grid, deltas, args.n_action, args.n_strategy)

    # Resolve output mode: 'auto' picks TEXT for models that parse native JSON-schema
    # poorly (gpt-oss), else native JSON_SCHEMA.
    if args.output_mode == "auto":
        output_mode = OutputMode.TEXT if "gpt-oss" in args.model.lower() else OutputMode.JSON_SCHEMA
    else:
        output_mode = OutputMode(args.output_mode)
    # TEXT-mode reasoning models (gpt-oss) emit a chain-of-thought channel *plus* the
    # answer's own (verbose, scaffolded) reasoning field; 2048 leaves room for both so
    # the trailing answer field (signal / prob_high) is not truncated.
    max_tokens = args.max_tokens or (2048 if output_mode == OutputMode.TEXT else 512)

    n = budget["TOTAL"]
    print(
        f"versions={versions}  cells={len(h_grid) * len(mu_grid) * len(deltas)} "
        f"(h={len(h_grid)} x mu={len(mu_grid)} x delta={len(deltas)})"
    )
    print(f"output_mode={output_mode.value}  max_tokens={max_tokens}  model={args.model}")
    print(f"calls per version: { {k: v for k, v in budget.items() if k != 'TOTAL'} }")
    print(f"TOTAL calls = {n}")
    for sec in (8.0, SEC_PER_CALL, 12.0):
        print(f"  est runtime @ {sec:.1f}s/call: {n * sec / 3600:.2f} h")
    if args.dry_run:
        return

    configure_logging()
    out = args.out or (
        Path(__file__).parent.parent.parent
        / "outputs"
        / "experiments"
        / f"strategy_map_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_map(
        versions,
        h_grid,
        mu_grid,
        deltas,
        args.n_action,
        args.n_strategy,
        args.temperature,
        args.model,
        out,
        output_mode=output_mode,
        max_tokens=max_tokens,
    )
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
