"""
Менеджер результатов симуляции.
Сохранение, загрузка, сравнение разных конфигураций.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResultsManager:
    """Управление результатами симуляций"""

    def __init__(self, storage_dir: str = "simulation_results"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.results_file = self.storage_dir / "results_index.json"
        self._load_index()

    def _load_index(self):
        """Загружает индекс существующих результатов"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except:
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Сохраняет индекс результатов"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def save_results(self, config: Dict, results: List[Dict], mode: str,
                     name: Optional[str] = None) -> str:
        """
        Сохраняет результаты симуляции с конфигом.

        Args:
            config: конфигурация симуляции
            results: список результатов для каждого SNR
            mode: 'random' или 'text'
            name: пользовательское имя (если None, генерируется автоматически)

        Returns:
            id результата
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{config['modulation']['type']}{config['modulation']['order']}_{mode}_{timestamp}"

        if name:
            result_id = f"{name}_{timestamp}"

        # Создаем папку для результатов
        result_dir = self.storage_dir / result_id
        result_dir.mkdir(exist_ok=True)

        # Сохраняем конфиг
        config_file = result_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Сохраняем результаты
        results_file = result_dir / "results.json"
        results_json = []
        for r in results:
            r_copy = r.copy()
            # Конвертируем numpy типы в стандартные
            for key in r_copy:
                if isinstance(r_copy[key], np.floating):
                    r_copy[key] = float(r_copy[key])
                elif isinstance(r_copy[key], np.integer):
                    r_copy[key] = int(r_copy[key])
            results_json.append(r_copy)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2)

        # Обновляем индекс
        self.index[result_id] = {
            "name": name or result_id,
            "timestamp": timestamp,
            "modulation": f"{config['modulation']['type']}-{config['modulation']['order']}",
            "coding": config['coding']['type'] if config['coding']['enabled'] else "none",
            "mode": mode,
            "snr_range": [float(min(config['ebn0_dB_range'])),
                          float(max(config['ebn0_dB_range']))],
            "num_points": len(results),
            "path": str(result_dir)
        }
        self._save_index()

        logger.info(f"Результаты сохранены: {result_id}")
        return result_id

    def load_results(self, result_id: str) -> Tuple[Dict, List[Dict]]:
        """
        Загружает результаты и конфиг по ID.

        Returns:
            (config, results)
        """
        result_dir = self.storage_dir / result_id

        with open(result_dir / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)

        with open(result_dir / "results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)

        return config, results

    def get_results_list(self) -> List[Dict]:
        """Возвращает список всех доступных результатов"""
        return list(self.index.values())

    def delete_results(self, result_id: str):
        """Удаляет результаты по ID"""
        import shutil
        if result_id in self.index:
            result_dir = Path(self.index[result_id]['path'])
            shutil.rmtree(result_dir, ignore_errors=True)
            del self.index[result_id]
            self._save_index()
            logger.info(f"Результаты удалены: {result_id}")

    def compare_results(self, result_ids: List[str]) -> Dict:
        """
        Сравнивает несколько результатов.

        Returns:
            Dict с данными для сравнения
        """
        comparison = {
            "results": [],
            "common_snr": None
        }

        all_configs = []
        all_results = []
        all_snr = []

        for rid in result_ids:
            config, results = self.load_results(rid)
            all_configs.append(config)
            all_results.append(results)
            snr_vals = [r['snr'] for r in results]
            all_snr.append(set(snr_vals))

            comparison["results"].append({
                "id": rid,
                "name": self.index[rid]['name'],
                "config": {
                    "modulation": f"{config['modulation']['type']}-{config['modulation']['order']}",
                    "coding": config['coding']['type'] if config['coding']['enabled'] else "none"
                },
                "data": results
            })

        # Найдем общие SNR для всех результатов
        if all_snr:
            common = all_snr[0]
            for s in all_snr[1:]:
                common = common.intersection(s)
            comparison["common_snr"] = sorted(list(common))

        return comparison

    def export_to_csv(self, result_id: str, filename: str):
        """Экспортирует результаты в CSV"""
        config, results = self.load_results(result_id)

        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        logger.info(f"Результаты экспортированы в: {filename}")