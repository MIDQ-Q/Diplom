"""
results_manager.py — Менеджер результатов симуляции.
Python 3.12+

Изменения (2 марта):
─────────────────────────────────────────────────────────────────────
_load_index() / _save_index():
  • Атомарная запись через временный файл + os.replace().
  • Корректная обработка json.JSONDecodeError.

save_results():
  • numpy-скаляры конвертируются через рекурсивный _to_python().
  • Гарантированно уникальный ID (суффикс _N при коллизии).
  • Новые поля в индексе:
      - coding_type (не только "none", но "hamming" / "ldpc" / "turbo")
      - has_per (bool)
      - has_rayleigh_theo (bool)
      - avg_encode_ms / avg_decode_ms  — среднее время encode/decode

load_results():
  • Явная обработка FileNotFoundError с информативным сообщением.

delete_results():
  • Проверка существования папки перед rmtree.
  • Возвращает bool.

rename_result():
  • Переименовывает запись в индексе (файлы не перемещаются).

compare_results():
  • Поддерживает разные SNR-сетки (пересечение + объединение).
  • Возвращает coding_gain_dB для каждого результата.

export_to_csv():
  • Принимает Path | str.
  • Расширенные поля: per, rayleigh_theoretical_ber, encode_time_ms, decode_time_ms.

export_comparison_csv():
  • Новый метод: экспорт сравнения нескольких результатов в один CSV.

get_summary():
  • Новый метод: краткая статистика по всем результатам (best BER, worst BER, и т.д.)
"""

import csv
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _to_python(obj: object) -> object:
    """
    Рекурсивно конвертирует numpy-скаляры и массивы в стандартные Python-типы.
    Необходим перед json.dump() для предотвращения TypeError.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


class ResultsManager:
    """Управление результатами симуляций: сохранение, загрузка, сравнение, экспорт."""

    def __init__(self, storage_dir: str = "simulation_results") -> None:
        self.storage_dir  = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.storage_dir / "results_index.json"
        self._load_index()

    # ── Индекс ────────────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Загружает индекс. При повреждённом JSON сбрасывает в {}."""
        if not self.results_file.exists():
            self.index: dict = {}
            return
        try:
            with open(self.results_file, encoding="utf-8") as f:
                self.index = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Индекс повреждён ({e}), сброс в пустой.")
            self.index = {}
        except OSError as e:
            logger.error(f"Ошибка чтения индекса: {e}")
            self.index = {}

    def _save_index(self) -> None:
        """
        Атомарная запись индекса через временный файл + os.replace().
        Если запись прерывается — старый файл остаётся нетронутым.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.storage_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.results_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save_results(self,
                     config: dict,
                     results: list[dict],
                     mode: str,
                     name: str | None = None) -> str:
        """
        Сохраняет результаты симуляции.

        Индекс обогащён новыми полями:
          coding_type       — "none" / "hamming" / "ldpc" / "turbo"
          has_per           — есть ли ненулевые PER
          has_rayleigh_theo — есть ли теоретические кривые Rayleigh
          avg_encode_ms     — среднее время encode
          avg_decode_ms     — среднее время decode

        Returns:
            Уникальный ID результата.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ct = (
            config["coding"]["type"] if config["coding"]["enabled"] else "none"
        )
        base_id = (
            f"{name}_{timestamp}" if name
            else f"{config['modulation']['type']}{config['modulation']['order']}_{ct}_{mode}_{timestamp}"
        )

        # Гарантируем уникальность ID
        result_id = base_id
        counter   = 1
        while result_id in self.index:
            result_id = f"{base_id}_{counter}"
            counter  += 1

        result_dir = self.storage_dir / result_id
        result_dir.mkdir(parents=True, exist_ok=True)

        with open(result_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(_to_python(config), f, indent=2, ensure_ascii=False)

        with open(result_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump([_to_python(r) for r in results], f, indent=2, ensure_ascii=False)

        rng             = config.get("ebn0_dB_range", [0, 0])
        has_per         = any(r.get("per", 0) > 0 for r in results)
        has_rayleigh    = any(r.get("rayleigh_theoretical_ber", 0) > 0 for r in results)
        enc_times       = [r.get("encode_time_ms", 0) for r in results if r.get("encode_time_ms")]
        dec_times       = [r.get("decode_time_ms", 0) for r in results if r.get("decode_time_ms")]

        self.index[result_id] = {
            "name":              name or result_id,
            "timestamp":         timestamp,
            "modulation":        f"{config['modulation']['type']}-{config['modulation']['order']}",
            "coding":            ct,
            "mode":              mode,
            "snr_range":         [float(min(rng)), float(max(rng))],
            "num_points":        len(results),
            "path":              str(result_dir),
            "has_per":           has_per,
            "has_rayleigh_theo": has_rayleigh,
            "avg_encode_ms":     float(np.mean(enc_times)) if enc_times else 0.0,
            "avg_decode_ms":     float(np.mean(dec_times)) if dec_times else 0.0,
        }
        self._save_index()
        logger.info(f"Результаты сохранены: {result_id}")
        return result_id

    def load_results(self, result_id: str) -> tuple[dict, list[dict]]:
        """
        Загружает (config, results) по ID.

        Raises:
            KeyError          — если ID не найден в индексе.
            FileNotFoundError — если папка на диске отсутствует.
        """
        if result_id not in self.index:
            raise KeyError(f"Результат не найден: {result_id!r}")
        result_dir = Path(self.index[result_id]["path"])
        if not result_dir.exists():
            raise FileNotFoundError(
                f"Папка результатов отсутствует: {result_dir}. "
                "Возможно, файлы были удалены вручную."
            )
        with open(result_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        with open(result_dir / "results.json", encoding="utf-8") as f:
            results = json.load(f)
        return config, results

    def delete_results(self, result_id: str) -> bool:
        """
        Удаляет результаты по ID.

        Returns:
            True — удалено; False — ID не найден.
        """
        if result_id not in self.index:
            logger.warning(f"Попытка удалить несуществующий ID: {result_id!r}")
            return False
        result_dir = Path(self.index[result_id]["path"])
        if result_dir.exists():
            shutil.rmtree(result_dir, ignore_errors=True)
        del self.index[result_id]
        self._save_index()
        logger.info(f"Результаты удалены: {result_id}")
        return True

    def rename_result(self, result_id: str, new_name: str) -> None:
        """Переименовывает запись в индексе (файлы не перемещаются)."""
        if result_id not in self.index:
            raise KeyError(f"Результат не найден: {result_id!r}")
        self.index[result_id]["name"] = new_name
        self._save_index()
        logger.info(f"Результат {result_id!r} переименован в {new_name!r}")

    # ── Запросы ───────────────────────────────────────────────────────────────

    def get_results_list(self) -> list[dict]:
        """Возвращает список записей индекса с полем 'id'."""
        return [{"id": rid, **info} for rid, info in self.index.items()]

    def get_summary(self) -> dict:
        """
        Краткая статистика по всем сохранённым результатам.

        Returns:
            {
                "total":           общее число результатов,
                "by_modulation":   {mod_str: count},
                "by_coding":       {coding_str: count},
                "best_ber":        {result_id, value},  # минимальный min_ber
                "with_per":        число результатов с данными PER,
                "avg_encode_ms":   среднее по всем сохранённым,
                "avg_decode_ms":   среднее по всем сохранённым,
            }
        """
        if not self.index:
            return {"total": 0}

        by_mod:  dict[str, int] = {}
        by_cod:  dict[str, int] = {}
        best_ber_id  = None
        best_ber_val = float("inf")
        with_per     = 0
        enc_ms_list: list[float] = []
        dec_ms_list: list[float] = []

        for rid, info in self.index.items():
            mod = info.get("modulation", "?")
            cod = info.get("coding", "none")
            by_mod[mod] = by_mod.get(mod, 0) + 1
            by_cod[cod] = by_cod.get(cod, 0) + 1
            if info.get("has_per", False):
                with_per += 1
            enc = info.get("avg_encode_ms", 0)
            dec = info.get("avg_decode_ms", 0)
            if enc > 0:
                enc_ms_list.append(enc)
            if dec > 0:
                dec_ms_list.append(dec)

            # Для best_ber нужно загружать данные — делаем только если индекс маленький
            if len(self.index) <= 50:
                try:
                    _, res = self.load_results(rid)
                    min_ber = min(r["ber"] for r in res if r["ber"] > 0) if res else float("inf")
                    if min_ber < best_ber_val:
                        best_ber_val = min_ber
                        best_ber_id  = rid
                except Exception:
                    pass

        return {
            "total":         len(self.index),
            "by_modulation": by_mod,
            "by_coding":     by_cod,
            "best_ber":      {"id": best_ber_id, "value": best_ber_val}
                             if best_ber_id else None,
            "with_per":      with_per,
            "avg_encode_ms": float(np.mean(enc_ms_list)) if enc_ms_list else 0.0,
            "avg_decode_ms": float(np.mean(dec_ms_list)) if dec_ms_list else 0.0,
        }

    # ── Сравнение ─────────────────────────────────────────────────────────────

    def compare_results(self, result_ids: list[str]) -> dict:
        """
        Сравнивает несколько результатов.

        Возвращает:
            {
                "results":        [...],  # данные каждого результата
                "common_snr":     [...],  # общие SNR-точки
                "all_snr_ranges": [...],  # объединение всех SNR-точек
                "coding_gains":   {id: float | nan},  # выигрыш кодирования
            }
        """
        comparison: dict = {
            "results":        [],
            "common_snr":     None,
            "all_snr_ranges": [],
            "coding_gains":   {},
        }
        all_snr_sets: list[set] = []

        for rid in result_ids:
            config, results = self.load_results(rid)
            snr_set = {r["snr"] for r in results}
            all_snr_sets.append(snr_set)

            # Coding gain для каждого результата
            ber_arr  = np.array([r["ber"]                    for r in results])
            theo_arr = np.array([r.get("theoretical_ber", 0) for r in results])
            snr_arr  = np.array([r["snr"]                    for r in results])
            try:
                from coding import compute_coding_gain
                gain = compute_coding_gain(theo_arr, ber_arr, snr_arr)
            except Exception:
                gain = float("nan")
            comparison["coding_gains"][rid] = float(gain)

            comparison["results"].append({
                "id":   rid,
                "name": self.index[rid]["name"],
                "config": {
                    "modulation": f"{config['modulation']['type']}-{config['modulation']['order']}",
                    "coding":     config["coding"]["type"] if config["coding"]["enabled"] else "none",
                },
                "data": results,
            })

        if all_snr_sets:
            common = all_snr_sets[0].copy()
            for s in all_snr_sets[1:]:
                common &= s
            union = set().union(*all_snr_sets)
            comparison["common_snr"]     = sorted(common)
            comparison["all_snr_ranges"] = sorted(union)

        return comparison

    # ── Экспорт ───────────────────────────────────────────────────────────────

    def export_to_csv(self, result_id: str,
                      filename: str | Path) -> Path:
        """
        Экспортирует результаты одного прогона в CSV.

        Формат файла:
          - Первые строки: метаданные в виде комментариев "# ключ: значение".
            Excel показывает их как текст; при импорте данных игнорирует.
          - Далее: заголовок и строки данных, разделитель — запятая,
            десятичный разделитель — точка (стандарт для Excel с любой локалью
            при открытии через Данные → Из текста/CSV).

        Поля данных (порядок столбцов):
            snr_dB, ber, ser, per,
            theoretical_ber, theoretical_ser, rayleigh_theoretical_ber,
            encode_time_ms, decode_time_ms,
            corrected_errors, detected_errors, total_blocks,
            num_bits_used, early_stop, adaptive_scale, spectral_efficiency

        Returns:
            Path к созданному файлу.

        Raises:
            KeyError / FileNotFoundError — если результат не найден.
            OSError — при ошибке записи файла.
        """
        config, results = self.load_results(result_id)
        out = Path(filename)

        # ── Метаданные для заголовка ─────────────────────────────────────────
        mod_cfg     = config.get("modulation", {})
        coding_cfg  = config.get("coding", {})
        ebn0_range  = config.get("ebn0_dB_range", [])

        modulation_str = f"{mod_cfg.get('type', '?')}-{mod_cfg.get('order', '?')}"
        if coding_cfg.get("enabled", False):
            ct = coding_cfg.get("type", "?")
            if ct == "turbo":
                coding_str = "Turbo (R≈1/3)"
            else:
                coding_str = (
                    f"{ct.upper()} "
                    f"({coding_cfg.get('n', '?')},{coding_cfg.get('k', '?')})"
                )
        else:
            coding_str = "нет"

        active_channels: set[str] = set()
        for r in results:
            active_channels.update(r.get("active_channels", []))
        channels_str = ", ".join(sorted(active_channels)) or "AWGN"

        snr_range_str = (
            f"{min(ebn0_range):.1f}..{max(ebn0_range):.1f} дБ"
            if len(ebn0_range) >= 2 else "?"
        )

        metadata_lines = [
            f"# Симулятор цифровой связи — экспорт результатов",
            f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Модуляция: {modulation_str}",
            f"# Кодирование: {coding_str}",
            f"# Каналы: {channels_str}",
            f"# Диапазон Eb/N0: {snr_range_str}",
            f"# Точек SNR: {len(results)}",
        ]

        # ── Нормализация строк данных ────────────────────────────────────────
        # Порядок столбцов фиксирован — совпадает с ТЗ
        fieldnames = [
            "snr_dB",
            "ber", "ber_pre_decrypt", "ber_post_decrypt",
            "ser", "per",
            "theoretical_ber", "theoretical_ser", "rayleigh_theoretical_ber",
            "encode_time_ms", "decode_time_ms",
            "encrypt_time_ms", "decrypt_time_ms",
            "corrected_errors", "detected_errors", "total_blocks",
            "num_bits_used", "early_stop", "adaptive_scale", "spectral_efficiency",
            "encryption_enabled", "cipher_name",
            "aes_block_errors", "error_propagation_factor",
        ]

        flat_results = []
        for r in results:
            flat_results.append({
                "snr_dB":                   r.get("snr",                      0),
                "ber":                      r.get("ber",                      0),
                "ber_pre_decrypt":          r.get("ber_pre_decrypt",          r.get("ber", 0)),
                "ber_post_decrypt":         r.get("ber_post_decrypt",         r.get("ber", 0)),
                "ser":                      r.get("ser",                      0),
                "per":                      r.get("per",                      0),
                "theoretical_ber":          r.get("theoretical_ber",          0),
                "theoretical_ser":          r.get("theoretical_ser",          0),
                "rayleigh_theoretical_ber": r.get("rayleigh_theoretical_ber", 0),
                "encode_time_ms":           r.get("encode_time_ms",           0),
                "decode_time_ms":           r.get("decode_time_ms",           0),
                "encrypt_time_ms":          r.get("encrypt_time_ms",          0),
                "decrypt_time_ms":          r.get("decrypt_time_ms",          0),
                "corrected_errors":         r.get("corrected_errors",         0),
                "detected_errors":          r.get("detected_errors",          0),
                "total_blocks":             r.get("total_blocks",             0),
                "num_bits_used":            r.get("num_bits_used",            0),
                "early_stop":               int(r.get("early_stop",           False)),
                "adaptive_scale":           r.get("adaptive_scale",           1),
                "spectral_efficiency":      r.get("spectral_efficiency",      0),
                "encryption_enabled":       int(r.get("encryption_enabled",   False)),
                "cipher_name":              r.get("cipher_name",              "none"),
                "aes_block_errors":         r.get("aes_block_errors",         0),
                "error_propagation_factor": r.get("error_propagation_factor", 1.0),
            })

        with open(out, "w", newline="", encoding="utf-8") as f:
            # Метаданные
            for line in metadata_lines:
                f.write(line + "\n")

            # Данные
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)

        logger.info(f"Результаты экспортированы в CSV: {out}")
        return out

    def export_comparison_csv(self, result_ids: list[str],
                               filename: str | Path) -> Path:
        """
        Экспортирует сравнение нескольких результатов в один CSV.

        Формат: одна строка на точку SNR для каждого результата.
        Столбцы: id, name, modulation, coding, snr, ber, ser, per,
                 theoretical_ber, rayleigh_theoretical_ber, coding_gain_dB.

        Returns:
            Path к созданному файлу.
        """
        comparison = self.compare_results(result_ids)
        out = Path(filename)
        rows: list[dict] = []

        for res in comparison["results"]:
            rid    = res["id"]
            gain   = comparison["coding_gains"].get(rid, float("nan"))
            for d in res["data"]:
                rows.append({
                    "id":                       rid,
                    "name":                     res["name"],
                    "modulation":               res["config"]["modulation"],
                    "coding":                   res["config"]["coding"],
                    "snr":                      d.get("snr",   0),
                    "ber":                      d.get("ber",   0),
                    "ser":                      d.get("ser",   0),
                    "per":                      d.get("per",   0),
                    "theoretical_ber":          d.get("theoretical_ber",   0),
                    "theoretical_ser":          d.get("theoretical_ser",   0),
                    "rayleigh_theoretical_ber": d.get("rayleigh_theoretical_ber", 0),
                    "coding_gain_dB":           gain if not np.isnan(gain) else "",
                })

        with open(out, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Сравнение экспортировано в CSV: {out}")
        return out
