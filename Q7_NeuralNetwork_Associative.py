#!/usr/bin/env python3
"""
Q7 Neural Network — tehnika: Quantum Associative Memory (Ventura-Martinez)
(čisto kvantno, bez klasičnog treniranja i bez hibrida).

Arhitektura:
  - Skladištenje „obrazaca“: svaka sedmorka iz CELOG CSV-a se mapira u nq-bitni hash
      h_mode="sum": h = (Σ num_i) mod 2^nq
      h_mode="xor": h = XOR_i ((num_i - 1) & mask_nq)
    Distribucija po hash stanjima = amplitude enkodiranje memorije:
      |ψ_mem⟩ = Σ_k √p_k |k⟩,  p_k = udeo redova čiji hash = k.
  - Dohvat: Grover-like amplifikacija TOP-M hash stanja po frekvenciji (marked).
    Difuzor = 2|ψ_mem⟩⟨ψ_mem| - I  (Ventura-Martinez varijanta oko memory state-a).
  - Readout: Statevector → bias_39 → NEXT rastuća sedmorka ∈ {1..39}.

Sve deterministički: seed=39; memorija i oracle iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, h_mode, M, Δiter) po meri cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import Operator, Statevector

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (6, 7, 8)
GRID_HMODE = ("sum", "xor")
GRID_M = (3, 5, 7, 10, 15)
GRID_ITER_DELTA = (-1, 0, 1)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


# =========================
# Hash + memory state
# =========================
def hash_row(row: np.ndarray, nq: int, mode: str) -> int:
    mask = (1 << nq) - 1
    if mode == "sum":
        return int(int(row.sum()) & mask)
    # xor
    h = 0
    for v in row:
        h ^= (int(v) - 1) & mask
    return int(h)


def memory_amplitudes(H: np.ndarray, nq: int, mode: str) -> np.ndarray:
    dim = 2 ** nq
    counts = np.zeros(dim, dtype=np.float64)
    for row in H:
        k = hash_row(np.asarray(row, dtype=int), nq, mode)
        counts[k] += 1.0
    s = float(counts.sum())
    if s <= 0.0:
        return np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    p = counts / s
    return np.sqrt(p)


def top_m_hash_states(mem_amps: np.ndarray, m: int) -> List[int]:
    p = mem_amps ** 2
    order = np.argsort(-p, kind="stable")
    # samo ne-nula verovatnoće
    nonzero = [int(i) for i in order if p[int(i)] > 0.0]
    return nonzero[:m]


# =========================
# Grover-like sa difuzorom oko |ψ_mem⟩
# =========================
def build_oracle(nq: int, marked: List[int]) -> QuantumCircuit:
    diag = np.ones(2 ** nq, dtype=complex)
    for k in marked:
        if 0 <= k < 2 ** nq:
            diag[k] = -1.0 + 0j
    return Diagonal(diag.tolist())


def build_mem_diffuser(nq: int, mem_amps: np.ndarray) -> Operator:
    """D = 2|ψ_mem⟩⟨ψ_mem| - I — refleksija oko memorijskog stanja."""
    v = mem_amps.astype(np.complex128).reshape(-1, 1)
    M = 2.0 * (v @ v.conj().T) - np.eye(v.shape[0], dtype=np.complex128)
    return Operator(M)


def overlap_squared(mem_amps: np.ndarray, marked: List[int]) -> float:
    """‖P_good |ψ_mem⟩‖^2 = Σ_{k∈marked} p_k."""
    p = mem_amps ** 2
    return float(sum(float(p[k]) for k in marked if 0 <= k < p.size))


def optimal_iterations_from_overlap(a2: float) -> int:
    """k* = round((π/(4θ)) - 1/2), sin(θ)=√a^2; minimum 1 iteracija."""
    a = float(np.sqrt(max(0.0, min(1.0, a2))))
    if a <= 1e-9 or a >= 1.0 - 1e-9:
        return 1
    theta = float(np.arcsin(a))
    k = int(round(np.pi / (4.0 * theta) - 0.5))
    return max(1, k)


def qam_probs(H: np.ndarray, nq: int, mode: str, m: int, k_iter: int) -> np.ndarray:
    mem = memory_amplitudes(H, nq, mode)
    marked = top_m_hash_states(mem, m)
    if not marked:
        p = mem ** 2
        s = float(p.sum())
        return p / s if s > 0 else p
    oracle = build_oracle(nq, marked)
    diff = build_mem_diffuser(nq, mem)

    qc = QuantumCircuit(nq)
    qc.initialize(mem.tolist(), range(nq))
    for _ in range(max(0, k_iter)):
        qc.compose(oracle, range(nq), inplace=True)
        qc.unitary(diff.data, range(nq), label="D_mem")

    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    s = float(p.sum())
    return p / s if s > 0 else p


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija po meri cos(bias, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    f_csv_n = f_csv / float(f_csv.sum() or 1.0)
    best = None
    for nq in GRID_NQ:
        for mode in GRID_HMODE:
            mem = memory_amplitudes(H, nq, mode)
            for M in GRID_M:
                marked = top_m_hash_states(mem, M)
                if not marked:
                    continue
                a2 = overlap_squared(mem, marked)
                k_star = optimal_iterations_from_overlap(a2)
                for d in GRID_ITER_DELTA:
                    k_iter = max(1, k_star + d)
                    try:
                        probs = qam_probs(H, nq, mode, M, k_iter)
                        b = bias_39(probs)
                        score = cosine(b, f_csv_n)
                    except Exception:
                        continue
                    key = (score, -nq, 0 if mode == "sum" else -1, -M, -abs(d))
                    if best is None or key > best[0]:
                        best = (
                            key,
                            dict(nq=nq, mode=mode, M=M, k_iter=k_iter, delta=d, a2=a2, score=score),
                        )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q7 NN (QAM — Quantum Associative Memory, Ventura-Martinez): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| hash:", best["mode"],
        "| M:", best["M"],
        "| iter:", best["k_iter"],
        "(Δ vs k*:", best["delta"], ")",
        "| overlap²:", round(float(best["a2"]), 6),
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    probs = qam_probs(H, best["nq"], best["mode"], best["M"], best["k_iter"])
    pred = pick_next_combination(probs)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q7 NN (QAM — Quantum Associative Memory, Ventura-Martinez): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 8 | hash: sum | M: 10 | iter: 3 (Δ vs k*: 1 ) | overlap²: 0.146087 | cos(bias, freq_csv): 0.995352
predikcija NEXT: (2, 5, 14, 23, 36, 37, 38)
"""



"""
Q7_NeuralNetwork_Associative.py — tehnika: QAM (Quantum Associative Memory, Ventura-Martinez)

Učita CEO CSV. Svaki red (sedmorku) mapira u nq-bitni hash preko sum ili xor moda.
Izgradi memorijsko stanje: |ψ_mem⟩ = Σ √p_k |k⟩, gde je p_k udeo redova čiji je hash = k.
Odredi „dobra“ stanja: TOP-M po verovatnoći p_k.
Primeni Grover-like amplifikaciju sa:
faznim oracle-om (Diagonal(±1) na marked),
memorijskim difuzorom D = 2|ψ_mem⟩⟨ψ_mem| - I (refleksija oko memorije, Ventura-Martinez).
Broj iteracija k* iz overlap² = Σ_{k∈marked} p_k (formula arcsin).
Statevector → bias_39 → NEXT.
Deterministička grid-optimizacija (nq, hash_mode, M, Δiter) po meri cos(bias, freq_csv).

Tehnike:
Skladištenje „obrazaca“ u kvantnu superpoziciju (amplitude-encoding distribucije hash-eva).
Ventura-Martinez dohvat: difuzor oko memorijskog, a ne oko uniformnog stanja.
Amplitude amplification sa deterministički izvedenim brojem iteracija.
Egzaktan Statevector.

Prednosti:
Prvi model u seriji koji koristi strukturu redova (celu sedmorku), a ne samo marginalu pojedinačnih brojeva.
Difuzor oko memorije razlikuje ovaj od običnog Grover-a (Q2) — model je „povezan“ sa podacima i u prvom i u drugom koraku.
Dva hash-a (sum, xor) grid-om se porede, bira se bolje.
Determinističko, brzo.

Nedostaci:
Hash kompresija je vrlo destruktivna: sedmorku 7 brojeva svodi na nq ≤ 8 bita → mnoge različite sedmorke mapiraju se u isto stanje; ovo uništava većinu strukture.
sum mod 2^nq je posebno grub (koncentriše mase oko proseka sume ~140 mod 256).
Memorijski difuzor preko Operator(matrix) daje 2^nq x 2^nq gustu matricu — nije kompatibilan sa realnim hardverom (čisto simulatorski).
„Query“ je opet top-M po frekvenciji — efektivno slično Grover-u; merilo cos(bias, freq_csv) je i ovde donekle tautološko.
Eksponencijalna cena u nq.
"""
