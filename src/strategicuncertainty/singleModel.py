import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from tqdm import tqdm

from src.strategicuncertainty.llm_interface import (
    load_template,
    query_llm,
)


@dataclass
class GameConfig:
    model_name: str = "openai/gpt-oss-20b"
    prompt_template_path: str = "prompt_templates/single_model_r1.j2"
    max_tokens: int = 256
    temperature: float = 0.0
    dataset_name: str = "watermelonhjg/TAL-SCQ5K-EN-R1"

    reward_utility: float = 1.0
    discount_factor: float = 0.9
    model_cost: float = 0.1

    num_trials: int = 10
    num_steps: int = 10
    output_dir: str = "outputs"
    seed: int = 42


# Functions below are just scaffolding for the implementation of the game
def ask_question(cfg: GameConfig, question: str):
    prompt = load_template(cfg.prompt_template_path, question=question)
    return query_llm(cfg.model_name, prompt, cfg.max_tokens, cfg.temperature)


def run_one_trial(
    cfg: GameConfig, dataset: datasets.Dataset, progress: Optional[tqdm] = None
) -> List[Dict]:
    results = []

    for step in range(cfg.num_steps):
        question = f"What is the best action at step {step}?"
        answer = ask_question(cfg, question)
        results.append({"step": step, "action": answer})
        if progress:
            progress.update(1)
            progress.refresh()
    return results


def run_trials(cfg: GameConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent.parent.parent
        / Path(cfg.output_dir)
        / f"single_model_{timestamp}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(cfg.dataset_name, split="test")

    progress = tqdm(total=cfg.num_trials * cfg.num_steps, desc="Running trials")
    for _ in range(cfg.num_trials):
        run_one_trial(cfg, dataset, progress)
    progress.close()

    with open(output_path / "metadata.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
