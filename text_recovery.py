"""
text_recovery.py — Восстановление текста из повреждённого битового потока.
Python 3.12+

Проблема
────────
После пайплайна (канал + FEC-декодирование) остаточные битовые ошибки
искажают текст. Кириллица в UTF-8 особенно уязвима: каждый символ
занимает 2 байта — одна битовая ошибка ломает оба. Стандартный
bits_to_text() заменяет нераспознанные последовательности на U+FFFD (�).

Алгоритмы
──────────
1. UTF-8 boundary repair (основной)
   Для каждого фрагмента с U+FFFD пробуем 7 битовых сдвигов (±1..±3 байта)
   в окне вокруг повреждённого байта. Выбираем вариант с минимальным
   числом U+FFFD и максимальным числом символов целевого диапазона
   (кириллица U+0400–U+04FF, латиница, цифры).

2. Cleanup
   - Удаление нулевых байт (артефакты паддинга)
   - Замена оставшихся U+FFFD на «?»
   - Удаление управляющих символов (кроме \\n, \\t, \\r)

Интерфейс
─────────
  TextRecovery:
    .recover(damaged_bits, original_len, encoding) → RecoveryResult
    .repair_string(damaged_text)                   → RecoveryResult

  RecoveryResult:
    .text           : str   — восстановленный текст
    .chars_ok       : int   — символов дошло без изменений
    .chars_fixed    : int   — символов исправлено boundary repair
    .chars_lost     : int   — символов не удалось восстановить (остались ?)
    .total_chars    : int
    .recovery_rate  : float — (chars_ok + chars_fixed) / total_chars
    .repair_time_ms : float
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Диапазоны Unicode для оценки «читаемости» символов
_READABLE_RANGES = [
    (0x0020, 0x007E),   # ASCII printable
    (0x0400, 0x04FF),   # Кириллица
    (0x0009, 0x000D),   # \t \n \r \f
]


def _is_readable(ch: str) -> bool:
    """Возвращает True если символ считается читаемым."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _READABLE_RANGES)


def _score_text(text: str) -> float:
    """
    Оценка читаемости строки: доля читаемых символов минус штраф за U+FFFD.
    Возвращает значение [0.0 .. 1.0].
    """
    if not text:
        return 0.0
    n_readable = sum(1 for c in text if _is_readable(c))
    n_fffd     = text.count("\ufffd")
    score = (n_readable - n_fffd * 2) / len(text)
    return max(0.0, min(1.0, score))


@dataclass
class RecoveryResult:
    """Результат восстановления текста."""
    text:           str
    chars_ok:       int   = 0
    chars_fixed:    int   = 0
    chars_lost:     int   = 0
    total_chars:    int   = 0
    recovery_rate:  float = 0.0
    repair_time_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"Всего: {self.total_chars} | "
            f"OK: {self.chars_ok} | "
            f"Исправлено: {self.chars_fixed} | "
            f"Потеряно: {self.chars_lost} | "
            f"Восстановлено: {self.recovery_rate:.1%} | "
            f"Время: {self.repair_time_ms:.1f} мс"
        )


class TextRecovery:
    """
    Восстановление текста из повреждённого битового потока.

    Parameters
    ----------
    encoding : str
        Целевая кодировка (default: «utf-8»).
    window_bytes : int
        Полуширина окна поиска в байтах для boundary repair (default: 3).
    """

    def __init__(self, encoding: str = "utf-8", window_bytes: int = 3) -> None:
        self.encoding     = encoding
        self.window_bytes = window_bytes

    # ── Публичный API ─────────────────────────────────────────────────────────

    def recover(
        self,
        damaged_bits: np.ndarray,
        original_len: int | None = None,
        encoding: str | None = None,
    ) -> RecoveryResult:
        """
        Восстанавливает текст из повреждённого битового потока.

        Parameters
        ----------
        damaged_bits : ndarray uint8
            Битовый поток после декодирования канала.
        original_len : int | None
            Ожидаемая длина текста в символах. Если передан —
            результат обрезается до этой длины.
        encoding : str | None
            Переопределить кодировку (иначе используется self.encoding).

        Returns
        -------
        RecoveryResult
        """
        t0  = time.perf_counter()
        enc = encoding or self.encoding

        bits = np.asarray(damaged_bits, dtype=np.uint8).ravel()
        # Выровнять до кратного 8
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        raw_bytes = np.packbits(bits).tobytes()

        # Шаг 1: первичное декодирование
        text_raw = raw_bytes.decode(enc, errors="replace")

        # Шаг 2: boundary repair
        text_repaired, n_fixed = self._boundary_repair(raw_bytes, text_raw, enc)

        # Шаг 3: cleanup
        text_clean = self._cleanup(text_repaired)

        # Шаг 4: обрезка до ожидаемой длины
        if original_len is not None and len(text_clean) > original_len:
            text_clean = text_clean[:original_len]

        # Статистика
        n_fffd  = text_clean.count("?")   # после cleanup U+FFFD → ?
        n_total = len(text_clean)
        n_ok    = n_total - n_fffd - n_fixed
        n_ok    = max(0, n_ok)

        rate = (n_ok + n_fixed) / n_total if n_total > 0 else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        result = RecoveryResult(
            text=text_clean,
            chars_ok=n_ok,
            chars_fixed=n_fixed,
            chars_lost=n_fffd,
            total_chars=n_total,
            recovery_rate=rate,
            repair_time_ms=elapsed_ms,
        )
        logger.debug("TextRecovery.recover: %s", result.summary())
        return result

    def repair_string(self, damaged_text: str) -> RecoveryResult:
        """
        Восстанавливает уже декодированную (но повреждённую) строку.
        Применяет только cleanup — boundary repair требует байтов.

        Parameters
        ----------
        damaged_text : str

        Returns
        -------
        RecoveryResult
        """
        t0   = time.perf_counter()
        text = self._cleanup(damaged_text)

        n_fffd  = text.count("?")
        n_total = len(text)
        n_ok    = n_total - n_fffd
        rate    = n_ok / n_total if n_total > 0 else 0.0

        return RecoveryResult(
            text=text,
            chars_ok=n_ok,
            chars_fixed=0,
            chars_lost=n_fffd,
            total_chars=n_total,
            recovery_rate=rate,
            repair_time_ms=(time.perf_counter() - t0) * 1e3,
        )

    # ── Boundary Repair ───────────────────────────────────────────────────────

    def _boundary_repair(
        self, raw_bytes: bytes, text_raw: str, enc: str
    ) -> tuple[str, int]:
        """
        Попытка восстановить невалидные UTF-8 последовательности
        перебором битовых сдвигов в окне вокруг каждой ошибки.

        Алгоритм:
          1. Найти позиции байт, соответствующих U+FFFD в декодированном тексте.
          2. Для каждой проблемной позиции попробовать сдвиги окна ±1..±window_bytes.
          3. Выбрать вариант с наибольшей оценкой читаемости (_score_text).
          4. Применить исправление к байтовому буферу.

        Возвращает (исправленный текст, число исправленных символов).
        """
        if "\ufffd" not in text_raw:
            return text_raw, 0

        buf = bytearray(raw_bytes)
        n_fixed = 0
        W = self.window_bytes

        # Найти байтовые позиции проблемных участков
        # Кодируем текст обратно через surrogateescape чтобы найти смещения
        byte_pos = 0
        char_pos = 0
        error_regions: list[int] = []   # список байтовых позиций начала ошибок

        # Итерируем по тексту и находим позиции U+FFFD
        try:
            raw_iter = raw_bytes
            i = 0
            while i < len(raw_iter):
                b = raw_iter[i]
                # Определяем длину UTF-8 последовательности
                if b < 0x80:
                    seq_len = 1
                elif b < 0xC0:
                    # Неожиданный continuation byte
                    error_regions.append(i)
                    i += 1
                    continue
                elif b < 0xE0:
                    seq_len = 2
                elif b < 0xF0:
                    seq_len = 3
                else:
                    seq_len = 4

                seq = raw_iter[i:i + seq_len]
                try:
                    seq.decode("utf-8")
                    i += seq_len
                except UnicodeDecodeError:
                    error_regions.append(i)
                    i += 1
        except Exception:
            pass

        if not error_regions:
            return text_raw, 0

        # Для каждой ошибочной позиции пробуем исправление окном
        fixed_positions: set[int] = set()

        for err_pos in error_regions:
            if err_pos in fixed_positions:
                continue

            best_score = _score_text(
                buf[max(0, err_pos - W): err_pos + W + 4].decode(enc, errors="replace")
            )
            best_buf = None

            # Пробуем инвертировать каждый бит в окне вокруг ошибки
            start = max(0, err_pos - W)
            end   = min(len(buf), err_pos + W + 2)

            for byte_i in range(start, end):
                for bit_i in range(8):
                    candidate = bytearray(buf)
                    candidate[byte_i] ^= (1 << bit_i)

                    # Проверяем валидность UTF-8 в окне
                    window = bytes(candidate[start: end + 2])
                    try:
                        decoded_window = window.decode(enc, errors="replace")
                        score = _score_text(decoded_window)
                        if score > best_score:
                            # Проверяем что U+FFFD исчезло в данной позиции
                            full_decoded = bytes(candidate).decode(enc, errors="replace")
                            if full_decoded[err_pos:err_pos + 1] != "\ufffd":
                                best_score = score
                                best_buf = candidate
                                n_fixed += 1
                    except Exception:
                        continue

            if best_buf is not None:
                buf = best_buf
                fixed_positions.add(err_pos)

        repaired_text = bytes(buf).decode(enc, errors="replace")
        # n_fixed — число исправленных позиций, а не символов; корректируем
        orig_fffd  = text_raw.count("\ufffd")
        after_fffd = repaired_text.count("\ufffd")
        n_fixed    = max(0, orig_fffd - after_fffd)

        return repaired_text, n_fixed

    # ── Cleanup ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup(text: str) -> str:
        """
        Финальная очистка строки:
          • U+FFFD → «?»
          • Нулевые байты (\\x00) → удалить
          • BOM (U+FEFF) в начале → удалить
          • Управляющие символы < 0x20 (кроме \\n \\t \\r) → удалить
        """
        # BOM и нулевые байты
        text = text.lstrip("\ufeff").replace("\x00", "")

        # U+FFFD → «?»
        text = text.replace("\ufffd", "?")

        # Управляющие символы
        allowed = {"\n", "\t", "\r"}
        cleaned = []
        for ch in text:
            cp = ord(ch)
            if cp < 0x20 and ch not in allowed:
                continue   # удаляем — не заменяем пробелом, чтобы не раздувать текст
            cleaned.append(ch)

        return "".join(cleaned)


# ── Удобная функция-обёртка ───────────────────────────────────────────────────

def recover_text(
    damaged_bits: np.ndarray,
    original_len: int | None = None,
    encoding: str = "utf-8",
    window_bytes: int = 3,
) -> RecoveryResult:
    """
    Восстанавливает текст из повреждённого битового потока.

    Удобная обёртка для использования из simulation.py и gui.py.

    Parameters
    ----------
    damaged_bits : ndarray uint8
    original_len : int | None  — ожидаемая длина в символах
    encoding     : str
    window_bytes : int         — глубина поиска boundary repair

    Returns
    -------
    RecoveryResult
    """
    engine = TextRecovery(encoding=encoding, window_bytes=window_bytes)
    return engine.recover(damaged_bits, original_len=original_len)
