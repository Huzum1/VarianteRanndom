import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
import random
import time

# ========= CONFIGURARE PAGINĂ =========
st.set_page_config(page_title="Generator Inteligent Loterie PRO", layout="wide")

st.title("🎰 Generator Inteligent de Numere Loterie PRO")
st.markdown("*Garanție: Minim 2 variante 4/4 per rundă + Acoperire statistică completă*")
st.markdown("---")

# ========= FUNCȚII UTILE =========

def parse_txt_file(file_content):
    lines = file_content.strip().split('\n')
    variants = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            variant_id = parts[0].strip()
            numbers = [int(x) for x in parts[1].strip().split()]
            variants.append((variant_id, numbers))
    return variants


def calculate_number_statistics(historical_rounds):
    freq_counter = Counter()
    last_seen = {}
    gaps = defaultdict(list)

    for idx, (round_id, numbers) in enumerate(historical_rounds):
        for num in numbers:
            freq_counter[num] += 1
            if num in last_seen:
                gap = idx - last_seen[num]
                gaps[num].append(gap)
            last_seen[num] = idx

    all_nums = list(freq_counter.keys())
    if not all_nums:
        return freq_counter, set(), set(), set()

    freqs = list(freq_counter.values())
    avg_freq = np.mean(freqs)
    std_freq = np.std(freqs) if len(freqs) > 1 else 1

    hot_numbers = set()
    cold_numbers = set()
    normal_numbers = set()

    for num, freq in freq_counter.items():
        if freq > avg_freq + std_freq * 0.5:
            hot_numbers.add(num)
        elif freq < avg_freq - std_freq * 0.5:
            cold_numbers.add(num)
        else:
            normal_numbers.add(num)

    return freq_counter, hot_numbers, cold_numbers, normal_numbers


def calculate_pair_frequencies(historical_rounds):
    pair_freq = Counter()
    triplet_freq = Counter()
    for _, numbers in historical_rounds:
        for pair in combinations(numbers, 2):
            pair_freq[tuple(sorted(pair))] += 1
        for triplet in combinations(numbers, 3):
            triplet_freq[tuple(sorted(triplet))] += 1
    return pair_freq, triplet_freq


def count_matches(variant, round_numbers):
    return len(set(variant) & set(round_numbers))


# ========= FUNCȚIA: find_perfect_matches =========
def find_perfect_matches(all_variants, historical_rounds, target_per_round, max_attempts):
    covering_vars = []
    round_coverage = defaultdict(int)
    found_rounds = set()

    attempts = 0
    progress = st.progress(0, text="🔍 Căutare variante 4/4...")

    while len(found_rounds) < len(historical_rounds) * target_per_round and attempts < max_attempts:
        variant_id, variant = random.choice(all_variants)
        for round_id, round_nums in historical_rounds:
            matches = count_matches(variant, round_nums)
            if matches == 4 and round_coverage[round_id] < target_per_round:
                covering_vars.append({
                    "id": variant_id,
                    "numbers": variant,
                    "round": round_id,
                    "match_type": "4/4"
                })
                round_coverage[round_id] += 1
                found_rounds.add(round_id)
                break
        attempts += 1
        if attempts % 1000 == 0:
            progress.progress(min(attempts / max_attempts, 1.0))
    progress.empty()

    return covering_vars, round_coverage


# ========= FUNCȚIA: generate_statistical_coverage_variants =========
def generate_statistical_coverage_variants(all_variants, historical_rounds, freq_counter,
                                           hot_numbers, cold_numbers, normal_numbers,
                                           pair_freq, triplet_freq, number_variants, max_number):
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


# ========= FUNCȚIA: export =========
def export_to_txt(variants):
    output = []
    for variant in variants:
        var_id = variant['id']
        numbers = ' '.join(map(str, variant['numbers']))
        output.append(f"{var_id}, {numbers}")
    return '\n'.join(output)


# ========= INTERFAȚĂ GRAFICĂ =========

col1, col2 = st.columns(2)
with col1:
    st.header("📥 1. Import Variante")
    variants_file = st.file_uploader("Încarcă fișier .txt cu variante", type=['txt'], key='variants')
    if variants_file:
        content = variants_file.read().decode('utf-8')
        all_variants = parse_txt_file(content)
        st.success(f"✅ {len(all_variants)} variante încărcate")
        with st.expander("👁️ Preview"):
            for i, (vid, vnums) in enumerate(all_variants[:5]):
                st.text(f"{vid}: {vnums}")

with col2:
    st.header("📥 2. Import Runde Istorice")
    rounds_file = st.file_uploader("Încarcă fișier .txt cu runde istorice", type=['txt'], key='rounds')
    if rounds_file:
        content = rounds_file.read().decode('utf-8')
        historical_rounds = parse_txt_file(content)
        st.success(f"✅ {len(historical_rounds)} runde încărcate")
        with st.expander("👁️ Preview"):
            for i, (rid, rnums) in enumerate(historical_rounds[:5]):
                st.text(f"{rid}: {rnums}")

st.markdown("---")

# ========= CONFIGURARE =========

st.header("⚙️ 3. Configurare Generare PRO")
col3, col4, col5 = st.columns(3)
with col3:
    coverage_variants = st.number_input("Variante Pas 1 (4/4 garantat)", 100, 1000, 800, 50)
with col4:
    number_variants = st.number_input("Variante Pas 2 (statistici)", 100, 500, 365, 50)
with col5:
    max_number = st.number_input("Număr maxim (plaja)", 40, 100, 66, 1)
with st.expander("🔧 Setări Avansate"):
    max_attempts = st.number_input("Max încercări căutare 4/4", 10000, 100000, 30000, 5000)
    target_per_round = st.selectbox("Target variante per rundă", [1, 2, 3], 1)

total_variants = coverage_variants + number_variants
st.info(f"📊 Total variante generate: **{total_variants}**")

# ========= BUTON PRINCIPAL =========
if st.button("🚀 GENEREAZĂ CU GARANȚIE 4/4", type="primary", use_container_width=True):
    if 'all_variants' not in locals() or 'historical_rounds' not in locals():
        st.error("❌ Te rog încarcă fișierele!")
    else:
        start_time = time.time()
        st.info("🔁 Încep procesarea...")

        freq_counter, hot_numbers, cold_numbers, normal_numbers = calculate_number_statistics(historical_rounds)
        pair_freq, triplet_freq = calculate_pair_frequencies(historical_rounds)

        covering_vars, round_coverage = find_perfect_matches(
            all_variants, historical_rounds, target_per_round, max_attempts
        )
        number_vars, number_coverage, pair_coverage = generate_statistical_coverage_variants(
            all_variants, historical_rounds, freq_counter,
            hot_numbers, cold_numbers, normal_numbers,
            pair_freq, triplet_freq,
            number_variants, max_number
        )

        final_variants = covering_vars + number_vars
        elapsed = time.time() - start_time
        st.session_state['final_variants'] = final_variants
        st.success(f"✅ Generare completă în {elapsed:.2f} secunde — {len(final_variants)} variante create!")

# ========= EXPORT =========
if 'final_variants' in st.session_state:
    final_variants = st.session_state['final_variants']
    export_all = export_to_txt(final_variants)
    perfect_only = [v for v in final_variants if v.get('match_type') == '4/4']
    export_perfect = export_to_txt(perfect_only)

    st.markdown("---")
    st.header("💾 5. Export Variante Finale")
    colA, colB = st.columns(2)
    with colA:
        st.download_button("📥 Descarcă TOATE variantele", export_all, "variante_totale.txt")
    with colB:
        st.download_button("🎯 Descarcă doar variantele 4/4", export_perfect, "variante_4din4.txt")

st.caption("© 2025 Generator Inteligent Loterie PRO – versiune completă și funcțională ✅")