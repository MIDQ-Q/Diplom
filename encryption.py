"""
encryption.py — Шифрование для симулятора цифровой связи.
Python 3.12+

Поддерживаемые алгоритмы
─────────────────────────
  XORCipher     — XOR с ключом (учебный потоковый шифр)
  AESCipher     — AES-128 в режимах ECB / CBC / CTR

Позиция в пайплайне
────────────────────
  данные → [шифр] → кодер → канал → декодер → [дешифр] → сравнение

Интерфейс
─────────
Все классы реализуют единый интерфейс:
  .encrypt(bits: ndarray) → ndarray    # uint8, длина ≥ len(bits)
  .decrypt(bits: ndarray) → ndarray    # uint8, длина = оригинал
  .name    → str
  .block_bits → int                    # 1 для XOR, 128 для AES

Особенности
───────────
• XORCipher: ключ циклически повторяется, нет padding, длина сохраняется.
• AES-ECB:   независимые 128-битные блоки, PKCS7-padding.
             Ошибка в блоке i → разрушает только блок i при дешифровке.
• AES-CBC:   блоки связаны через IV. Ошибка в блоке i → разрушает блок i
             и один бит блока i+1 (лавинный эффект).
             IV (16 байт) prepend к шифртексту.
• AES-CTR:   потоковый режим через счётчик. Ошибки не распространяются.
             Nonce (8 байт) prepend к шифртексту.
• Для всех AES-режимов оригинальная длина битов хранится в первых 4 байтах
  шифртекста (little-endian uint32) чтобы корректно снять padding после
  дешифровки в условиях битовых ошибок.
"""

import os
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Определяем бэкенд AES ────────────────────────────────────────────────────
_AES_BACKEND: str = "none"

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    _AES_BACKEND = "cryptography"
    logger.debug("AES бэкенд: cryptography")
except ImportError:
    pass

if _AES_BACKEND == "none":
    try:
        from Crypto.Cipher import AES as _PycAES
        _AES_BACKEND = "pycryptodome"
        logger.debug("AES бэкенд: pycryptodome")
    except ImportError:
        pass

if _AES_BACKEND == "none":
    logger.warning(
        "Библиотеки AES не найдены (cryptography, pycryptodome). "
        "Доступен только XOR-шифр. Установите: pip install cryptography"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """uint8 ndarray бит → bytes. Длина должна быть кратна 8."""
    b = np.packbits(bits.astype(np.uint8))
    return b.tobytes()


def _bytes_to_bits(data: bytes, n_bits: Optional[int] = None) -> np.ndarray:
    """bytes → uint8 ndarray бит. Если n_bits задан — обрезает по нему."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    if n_bits is not None:
        bits = bits[:n_bits]
    return bits.astype(np.uint8)


def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    """PKCS7 padding до кратности block_size байт."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def _pkcs7_unpad(data: bytes) -> bytes:
    """Снятие PKCS7 padding. При ошибке возвращает data без изменений."""
    if not data:
        return data
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16:
        return data
    return data[:-pad_len]


# ══════════════════════════════════════════════════════════════════════════════
# XOR-шифр
# ══════════════════════════════════════════════════════════════════════════════

class XORCipher:
    """
    Учебный XOR-шифр (потоковый).

    Ключ циклически XOR-ится с данными побайтово.
    Не требует padding. Ошибки не распространяются.
    Служит baseline: BER_post_decrypt == BER_pre_decrypt.

    Parameters
    ----------
    key_hex : hex-строка ключа (любая длина, рекомендуется 16+ байт).
              None → генерируется os.urandom(16).
    """

    block_bits = 8  # побайтовая работа

    def __init__(self, key_hex: Optional[str] = None) -> None:
        if key_hex:
            try:
                key_bytes = bytes.fromhex(key_hex.replace(" ", ""))
            except ValueError:
                logger.warning("XOR: неверный hex-ключ, генерируем случайный")
                key_bytes = os.urandom(16)
        else:
            key_bytes = os.urandom(16)

        if len(key_bytes) == 0:
            key_bytes = os.urandom(16)

        self._key = np.frombuffer(key_bytes, dtype=np.uint8)
        self.name = f"XOR-{len(key_bytes)*8}bit"
        logger.debug("XORCipher инициализирован, ключ %d байт", len(key_bytes))

    @property
    def key_hex(self) -> str:
        return self._key.tobytes().hex()

    def encrypt(self, bits: np.ndarray) -> np.ndarray:
        """XOR биты с циклически повторяющимся ключом."""
        t0 = time.perf_counter()
        bits = np.asarray(bits, dtype=np.uint8).ravel()

        # Расширяем ключ до длины данных (побайтово → побитово)
        n_bytes = (len(bits) + 7) // 8
        key_repeated = np.tile(self._key, (n_bytes // len(self._key)) + 1)[:n_bytes]
        key_bits = np.unpackbits(key_repeated)[:len(bits)]

        result = (bits ^ key_bits).astype(np.uint8)
        logger.debug("XOR encrypt: %d bits  (%.3f ms)",
                     len(bits), (time.perf_counter() - t0) * 1e3)
        return result

    def decrypt(self, bits: np.ndarray) -> np.ndarray:
        """XOR симметричен — decrypt == encrypt."""
        return self.encrypt(bits)


# ══════════════════════════════════════════════════════════════════════════════
# AES-шифр (ECB / CBC / CTR)
# ══════════════════════════════════════════════════════════════════════════════

class AESCipher:
    """
    AES-128 в режимах ECB, CBC, CTR.

    Пайплайн encrypt()
    ──────────────────
    1. bits → bytes (packbits)
    2. Prepend length header (4 байта, little-endian): число бит оригинала
    3. PKCS7 padding до кратности 16 байт (только ECB/CBC; CTR без padding)
    4. AES encrypt
    5. Для CBC: prepend IV (16 байт)
       Для CTR: prepend nonce (8 байт)
    6. bytes → bits

    Пайплайн decrypt()
    ──────────────────
    1. bits → bytes
    2. Для CBC: извлечь IV из начала; для CTR: nonce
    3. AES decrypt
    4. PKCS7 unpad (ECB/CBC)
    5. Извлечь length header → обрезать до оригинальной длины
    6. bytes → bits

    Распространение ошибок по режимам
    ──────────────────────────────────
    ECB: ошибка в блоке i → разрушает только блок i (16 байт)
    CBC: ошибка в блоке i → разрушает блок i полностью + 1 бит в блоке i+1
    CTR: ошибки не распространяются (потоковый режим)

    Parameters
    ----------
    mode    : "ECB" | "CBC" | "CTR"
    key_hex : 32 hex-символа (128 бит). None → os.urandom(16).
    """

    block_bits  = 128
    HEADER_BYTES = 4   # uint32 little-endian: длина исходных бит

    def __init__(self, mode: str = "CBC", key_hex: Optional[str] = None) -> None:
        mode = mode.upper()
        if mode not in ("ECB", "CBC", "CTR"):
            raise ValueError(f"AES: неизвестный режим {mode!r}. "
                             f"Допустимо: 'ECB', 'CBC', 'CTR'.")
        if _AES_BACKEND == "none":
            raise RuntimeError(
                "AES недоступен: установите 'cryptography' или 'pycryptodome'."
            )

        self.mode = mode
        self.name = f"AES-128-{mode}"

        if key_hex:
            try:
                key_bytes = bytes.fromhex(key_hex.replace(" ", ""))
                if len(key_bytes) != 16:
                    raise ValueError
            except ValueError:
                logger.warning("AES: неверный ключ (%s), генерируем случайный", key_hex)
                key_bytes = os.urandom(16)
        else:
            key_bytes = os.urandom(16)

        self._key = key_bytes
        logger.debug("AESCipher(%s) инициализирован", mode)

    @property
    def key_hex(self) -> str:
        return self._key.hex()

    # ── Внутренние методы шифрования (зависят от бэкенда) ────────────────────

    def _aes_encrypt_raw(self, plaintext: bytes, iv_or_nonce: bytes) -> bytes:
        """Шифрует plaintext, возвращает только шифртекст (без IV/nonce)."""
        if _AES_BACKEND == "cryptography":
            if self.mode == "ECB":
                cipher = Cipher(algorithms.AES(self._key),
                                modes.ECB(),
                                backend=default_backend())
            elif self.mode == "CBC":
                cipher = Cipher(algorithms.AES(self._key),
                                modes.CBC(iv_or_nonce),
                                backend=default_backend())
            else:  # CTR
                cipher = Cipher(algorithms.AES(self._key),
                                modes.CTR(iv_or_nonce),
                                backend=default_backend())
            enc = cipher.encryptor()
            return enc.update(plaintext) + enc.finalize()

        else:  # pycryptodome
            if self.mode == "ECB":
                c = _PycAES.new(self._key, _PycAES.MODE_ECB)
            elif self.mode == "CBC":
                c = _PycAES.new(self._key, _PycAES.MODE_CBC, iv=iv_or_nonce)
            else:
                c = _PycAES.new(self._key, _PycAES.MODE_CTR,
                                nonce=iv_or_nonce[:8])
            return c.encrypt(plaintext)

    def _aes_decrypt_raw(self, ciphertext: bytes, iv_or_nonce: bytes) -> bytes:
        """Дешифрует ciphertext, возвращает plaintext."""
        if _AES_BACKEND == "cryptography":
            if self.mode == "ECB":
                cipher = Cipher(algorithms.AES(self._key),
                                modes.ECB(),
                                backend=default_backend())
            elif self.mode == "CBC":
                cipher = Cipher(algorithms.AES(self._key),
                                modes.CBC(iv_or_nonce),
                                backend=default_backend())
            else:
                cipher = Cipher(algorithms.AES(self._key),
                                modes.CTR(iv_or_nonce),
                                backend=default_backend())
            dec = cipher.decryptor()
            return dec.update(ciphertext) + dec.finalize()

        else:  # pycryptodome
            if self.mode == "ECB":
                c = _PycAES.new(self._key, _PycAES.MODE_ECB)
            elif self.mode == "CBC":
                c = _PycAES.new(self._key, _PycAES.MODE_CBC, iv=iv_or_nonce)
            else:
                c = _PycAES.new(self._key, _PycAES.MODE_CTR,
                                nonce=iv_or_nonce[:8])
            return c.decrypt(ciphertext)

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def encrypt(self, bits: np.ndarray) -> np.ndarray:
        """
        Шифрует битовый массив.

        Структура выходного шифртекста (в байтах, до преобразования в биты):
          [4 байта: длина bits] [IV/nonce, если CBC/CTR] [AES-шифртекст]
        """
        t0 = time.perf_counter()
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        n_bits_orig = len(bits)

        # Биты → байты с padding до кратности 8
        pad8 = (-n_bits_orig) % 8
        if pad8:
            bits_padded = np.concatenate([bits, np.zeros(pad8, dtype=np.uint8)])
        else:
            bits_padded = bits
        plaintext = _bits_to_bytes(bits_padded)

        # Prepend length header
        header = n_bits_orig.to_bytes(self.HEADER_BYTES, "little")
        plaintext = header + plaintext

        if self.mode in ("ECB", "CBC"):
            plaintext = _pkcs7_pad(plaintext, 16)

        # Генерируем IV / nonce
        if self.mode == "CBC":
            iv = os.urandom(16)
            ciphertext = self._aes_encrypt_raw(plaintext, iv)
            output_bytes = iv + ciphertext
        elif self.mode == "CTR":
            # CTR в cryptography требует 16-байтный nonce; храним 8 + 8 нулей
            nonce8 = os.urandom(8)
            nonce16 = nonce8 + b"\x00" * 8
            ciphertext = self._aes_encrypt_raw(plaintext, nonce16)
            output_bytes = nonce8 + ciphertext
        else:  # ECB
            ciphertext = self._aes_encrypt_raw(plaintext, b"")
            output_bytes = ciphertext

        result = _bytes_to_bits(output_bytes)
        logger.debug("AES-%s encrypt: %d → %d bits  (%.3f ms)",
                     self.mode, n_bits_orig, len(result),
                     (time.perf_counter() - t0) * 1e3)
        return result

    def decrypt(self, bits: np.ndarray) -> np.ndarray:
        """
        Дешифрует битовый массив.

        Возвращает массив ровно той длины, что был подан в encrypt().
        При повреждении данных (битовые ошибки канала) пытается максимально
        корректно извлечь данные, не вызывая исключений.
        """
        t0 = time.perf_counter()
        bits = np.asarray(bits, dtype=np.uint8).ravel()

        # Биты → байты
        pad8 = (-len(bits)) % 8
        if pad8:
            bits = np.concatenate([bits, np.zeros(pad8, dtype=np.uint8)])
        raw = _bits_to_bytes(bits)

        try:
            if self.mode == "CBC":
                if len(raw) < 16 + self.HEADER_BYTES + 16:
                    raise ValueError("Слишком короткий шифртекст для CBC")
                iv          = raw[:16]
                ciphertext  = raw[16:]
                # Выравниваем до кратности 16 для AES
                trim = len(ciphertext) - (len(ciphertext) % 16)
                if trim < 16:
                    raise ValueError("Нет полного блока шифртекста")
                ciphertext = ciphertext[:trim]
                plaintext  = self._aes_decrypt_raw(ciphertext, iv)
                plaintext  = _pkcs7_unpad(plaintext)

            elif self.mode == "CTR":
                if len(raw) < 8 + self.HEADER_BYTES:
                    raise ValueError("Слишком короткий шифртекст для CTR")
                nonce8    = raw[:8]
                nonce16   = nonce8 + b"\x00" * 8
                ciphertext = raw[8:]
                plaintext  = self._aes_decrypt_raw(ciphertext, nonce16)

            else:  # ECB
                trim = len(raw) - (len(raw) % 16)
                if trim < 16:
                    raise ValueError("Нет полного блока шифртекста для ECB")
                plaintext = self._aes_decrypt_raw(raw[:trim], b"")
                plaintext = _pkcs7_unpad(plaintext)

            # Извлекаем length header
            if len(plaintext) < self.HEADER_BYTES:
                raise ValueError("plaintext короче заголовка длины")
            n_bits_orig = int.from_bytes(
                plaintext[:self.HEADER_BYTES], "little"
            )
            payload = plaintext[self.HEADER_BYTES:]
            result  = _bytes_to_bits(payload, n_bits_orig)

        except Exception as e:
            # При ошибке дешифровки возвращаем нули длиной входа
            # (это честно отражает полную потерю блока)
            logger.debug("AES decrypt error: %s — возвращаем нули", e)
            n_bits_orig = len(bits) - (
                16 * 8 if self.mode == "CBC" else
                8  * 8 if self.mode == "CTR" else 0
            ) - self.HEADER_BYTES * 8
            n_bits_orig = max(n_bits_orig, 8)
            result = np.zeros(n_bits_orig, dtype=np.uint8)

        logger.debug("AES-%s decrypt: %d bits  (%.3f ms)",
                     self.mode, len(result),
                     (time.perf_counter() - t0) * 1e3)
        return result.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Фабрика
# ══════════════════════════════════════════════════════════════════════════════

def get_cipher(cipher_type: str,
               mode: str = "CBC",
               key_hex: Optional[str] = None):
    """
    Фабричный метод для создания шифра.

    Parameters
    ----------
    cipher_type : "xor" | "aes" | "none"
    mode        : режим AES ("ECB", "CBC", "CTR"); игнорируется для XOR
    key_hex     : hex-строка ключа; None → os.urandom

    Returns
    -------
    XORCipher | AESCipher | None
    """
    ct = cipher_type.lower().strip()
    if ct == "none" or ct == "":
        return None
    if ct == "xor":
        return XORCipher(key_hex=key_hex)
    if ct == "aes":
        return AESCipher(mode=mode, key_hex=key_hex)
    raise ValueError(f"Неизвестный тип шифра: {cipher_type!r}. "
                     f"Допустимо: 'xor', 'aes', 'none'.")


def aes_available() -> bool:
    """True если AES-бэкенд доступен."""
    return _AES_BACKEND != "none"


# ══════════════════════════════════════════════════════════════════════════════
# Аналитика ошибок после дешифровки
# ══════════════════════════════════════════════════════════════════════════════

def compute_encryption_stats(
    original_bits:   np.ndarray,
    decrypted_bits:  np.ndarray,
    decoded_bits:    np.ndarray,
    cipher,
) -> dict:
    """
    Вычисляет статистику ошибок с учётом шифрования.

    Parameters
    ----------
    original_bits  : исходные данные до шифрования
    decrypted_bits : данные после декодера + дешифровки
    decoded_bits   : данные после декодера (до дешифровки, зашифрованные)
    cipher         : объект шифра (XORCipher | AESCipher) или None

    Returns
    -------
    dict: ber_post_decrypt, aes_block_errors, error_propagation_factor
    """
    n = min(len(original_bits), len(decrypted_bits))
    if n == 0:
        return {
            "ber_post_decrypt":         0.0,
            "aes_block_errors":         0,
            "error_propagation_factor": 1.0,
        }

    bit_errors_post = int(np.sum(
        original_bits[:n] != decrypted_bits[:n]
    ))
    ber_post = bit_errors_post / n

    # BER до дешифровки (на зашифрованных данных)
    n2 = min(len(original_bits), len(decoded_bits))
    # decoded_bits содержит зашифрованный поток с ошибками канала
    # для сравнения используем encrypted_bits (те что были до канала)
    # Если cipher None — ber_pre == ber_post
    ber_pre = ber_post  # будет перезаписан снаружи если нужно

    # Подсчёт повреждённых AES-блоков (только для AES)
    aes_block_errors = 0
    if cipher is not None and isinstance(cipher, AESCipher):
        block_b = 16  # 16 байт = 128 бит
        n_blocks = (n + block_b * 8 - 1) // (block_b * 8)
        for i in range(n_blocks):
            s = i * block_b * 8
            e = min(s + block_b * 8, n)
            if np.any(original_bits[s:e] != decrypted_bits[s:e]):
                aes_block_errors += 1

    epf = (ber_post / ber_pre) if ber_pre > 1e-12 else 1.0

    return {
        "ber_post_decrypt":         float(ber_post),
        "aes_block_errors":         aes_block_errors,
        "error_propagation_factor": float(epf),
    }
