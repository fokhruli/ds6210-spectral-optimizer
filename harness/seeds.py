"""Cryptographic seed derivation.  §7 of the handout.

Each student is assigned three seeds via

    s_i = SHA256(student_id || course_salt || i) mod 2^31

These seeds must anchor ALL reported plots and logs.  Use the same
three values across every task and every optimizer.
"""
from __future__ import annotations

import hashlib


COURSE_SALT_DEFAULT = "ds6210-spring2026"


def derive_seed(student_id: str, course_salt: str, i: int) -> int:
    payload = f"{student_id}||{course_salt}||{i}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def derive_seeds(student_id: str, course_salt: str = COURSE_SALT_DEFAULT,
                 n: int = 3) -> list[int]:
    return [derive_seed(student_id, course_salt, i) for i in range(1, n + 1)]


if __name__ == "__main__":
    import sys
    student_id = sys.argv[1] if len(sys.argv) > 1 else "jdoe"
    seeds = derive_seeds(student_id, n=3)
    print(f"student_id = {student_id}")
    print(f"course_salt = {COURSE_SALT_DEFAULT}")
    print(f"three seeds = {seeds}")
