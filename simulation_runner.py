"""
simulation_runner.py — тонкий слой “запускатель” + хранилище истории результатов.

Задача:
  - убрать глобальный results_manager из simulation.py (ядро симуляции должно быть чистым)
  - дать GUI единое место, откуда брать results_manager и запускать симуляции
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from results_manager import ResultsManager
from simulation import simulate_transmission, simulate_text_transmission


results_manager = ResultsManager()


@dataclass
class SimulationRunner:
    manager: ResultsManager = results_manager

    def run_random_point(self, config: dict, ebn0_dB: float, *, seed: int | None = None, prev_ber: float | None = None) -> dict[str, Any]:
        if seed is not None:
            np.random.seed(int(seed))
        return simulate_transmission(config, ebn0_dB=float(ebn0_dB), prev_ber=prev_ber, log_config_once=False)

    def run_text_point(self, config: dict, text: str, ebn0_dB: float, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            np.random.seed(int(seed))
        return simulate_text_transmission(config, text=text, ebn0_dB=float(ebn0_dB), log_config_once=False)

