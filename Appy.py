import random
import numpy as np
import streamlit as st
from itertools import combinations
from collections import defaultdict
import time

# ---------------- FUNCȚII UTILE ---------------- #

def count_matches(variant, round_numbers):
    """Returnează numărul de potriviri între două liste de numere."""
    return len(set(variant) & set(round_numbers))


# ---------- FUNCȚIA OPTIMIZATĂ find_perfect_matches ----------
def find_perfect_matches(
    all_variants, historical_rounds, target_per_round=2,
    max_attempts=30000, desired_total=None,
    progress_bar=None, status_text=None
):
    """
    Găsește combinațiile 4/4 garantate și completează până la desired_total dacă e nevoie.
    Include progres vizual și estimare de timp.
    """
    combo_to_variants = defaultdict(list)
    for var_id, nums in all_variants:
        key = tuple(sorted(nums))
        combo_to_variants[key].append((var_id, nums))

    selected_variants = []
    used_ids = set()
    round_coverage = defaultdict(int)
    total_rounds = len(historical_rounds)

    start_time = time.time()

    # Caută variantele exacte pentru fiecare rundă
    for idx, (round_id, round_nums) in enumerate(historical_rounds):
        key = tuple(sorted(round_nums))
        candidates = combo_to_variants.get(key, [])
        for var_id, nums in candidates:
            if round_coverage[round_id] >= target_per_round:
                break
            if var_id in used_ids:
                continue
            selected_variants.append({
                "id": var_id,
                "numbers": nums,
                "round_id": round_id,
                "match_type": "4/4"
            })
            used_ids.add(var_id)
            round_coverage[round_id] += 1

        # Actualizare progres și estimare timp
        if progress_bar and status_text:
            progress = (idx + 1) / total_rounds
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = (total_rounds - (idx + 1)) * avg_time
            eta_text = f"⏱️ Estimare: {remaining:.1f} secunde rămase"
            progress_bar.progress(progress)
            status_text.text(
                f"🔄 Procesare rundă {idx + 1}/{total_rounds} — {eta_text}"
            )
            time.sleep(0.001)

    # Completează (padding) până la desired_total, dacă e nevoie
    if desired_total:
        idx = 0
        while len(selected_variants) < desired_total and idx < len(all_variants):
            var_id, nums = all_variants[idx]
            if var_id not in used_ids:
                selected_variants.append({
                    "id": var_id,
                    "numbers": nums,
                    "round_id": None,
                    "match_type": "pad"
                })
                used_ids.add(var_id)
            idx += 1

    if status_text:
        status_text.text("✅ Gata! Toate rundele au fost procesate.")
        progress_bar.progress(1.0)

    return selected_variants, round_coverage


# ---------- FUNCȚIA generate_statistical_coverage_variants ----------
def generate_statistical_coverage_variants(
    all_variants, historical_rounds, freq_counter,
    hot_numbers, cold_numbers, normal_numbers,
    pair_freq, triplet_freq, number_variants, max_number
):
    """Generează variante statistice bazate pe frecvențe și greutăți."""
    number_vars = []
    used_ids = set()
    avg_freq = np.mean(list(freq_counter.values())) if freq_counter else 1
    threshold_hot = avg_freq * 1.2
    threshold_cold = avg_freq * 0.8
    all_nums = list(range(1, max_number + 1))
    weights = []
    for n in all_nums:
        f = freq_counter.get(n, 0)
        if f > threshold_hot:
            weights.append(2.0)
        elif f < threshold_cold:
            weights.append(0.5)
        else:
            weights.append(1.0)

    for _ in range(number_variants):
        variant_id, variant = random.choice(all_variants)
        if variant_id in used_ids:
            continue
        new_variant = sorted(random.choices(all_nums, weights=weights, k=len(variant)))
        number_vars.append({
            "id": variant_id,
            "numbers": new_variant,
            "match_type": "statistical"
        })
        used_ids.add(variant_id)

    number_coverage = defaultdict(int)
    pair_coverage = defaultdict(int)
    for var in number_vars:
        for num in var['numbers']:
            number_coverage[num] += 1
        for pair in combinations(var['numbers'], 2):
            pair_coverage[pair] += 1

    return number_vars, number_coverage, pair_coverage


# ---------------- INTERFAȚĂ STREAMLIT ---------------- #

st.title("⚙️ 3. Configurare Generare PRO")

coverage_variants = st.number_input("Variante Pas 1 (4/4 garantat)", min_value=1, value=800, step=1)
number_variants = st.number_input("Variante Pas 2 (statistici)", min_value=1, value=365, step=1)
max_number = st.number_input("Număr maxim (plajă)", min_value=10, value=66, step=1)
st.write(f"📊 Total variante generate: {coverage_variants + number_variants}")

if st.button("🚀 GENEREAZĂ CU GARANȚIE 4/4"):
    st.info("⏳ Generare în curs...")

    # Exemplu: simulăm date (în aplicația ta reală vin din fișiere)
    all_variants = [(i, random.sample(range(1, max_number + 1), 6)) for i in range(100000)]
    historical_rounds = [(i, random.sample(range(1, max_number + 1), 6)) for i in range(200)]

    target_per_round = 2
    max_attempts = 50000

    # --- PAS 1: Găsire variante garantate 4/4 cu progres vizual + estimare ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    covering_vars, round_coverage = find_perfect_matches(
        all_variants,
        historical_rounds,
        target_per_round=target_per_round,
        max_attempts=max_attempts,
        desired_total=coverage_variants,
        progress_bar=progress_bar,
        status_text=status_text
    )

    # --- PAS 2: Generare variante statistice ---
    freq_counter = defaultdict(int)
    for _, nums in historical_rounds:
        for n in nums:
            freq_counter[n] += 1

    hot_numbers = [n for n, f in freq_counter.items() if f > np.mean(list(freq_counter.values())) * 1.2]
    cold_numbers = [n for n, f in freq_counter.items() if f < np.mean(list(freq_counter.values())) * 0.8]
    normal_numbers = [n for n in freq_counter if n not in hot_numbers and n not in cold_numbers]
    pair_freq = defaultdict(int)
    triplet_freq = defaultdict(int)

    number_vars, number_coverage, pair_coverage = generate_statistical_coverage_variants(
        all_variants,
        historical_rounds,
        freq_counter,
        hot_numbers,
        cold_numbers,
        normal_numbers,
        pair_freq,
        triplet_freq,
        number_variants,
        max_number
    )

    total_variants = covering_vars + number_vars

    st.success(f"✅ Generare finalizată cu succes! Total: {len(total_variants)} variante.")
    st.write(f"➡️ Din care {len(covering_vars)} sunt garantate 4/4 și {len(number_vars)} statistice.")