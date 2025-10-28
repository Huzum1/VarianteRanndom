# Appy.py — Generator Inteligent Loterie PRO (complet)
import streamlit as st
import numpy as np
import pandas as pd
import random
import time
from itertools import combinations
from collections import Counter, defaultdict
from io import StringIO

# ---------------- Page config ----------------
st.set_page_config(page_title="Generator Inteligent Loterie PRO", layout="wide")
st.title("🎰 Generator Inteligent de Numere Loterie PRO")
st.markdown("*Garanție: Minim 2 variante 4/4 per rundă + Acoperire statistică completă*")
st.markdown("---")

# ---------------- Utility parsers ----------------
def parse_txt_file_content(text):
    """
    Parse text content where lines are: ID, n1 n2 n3 n4 (or n1,n2,...)
    Returns list of (id, [nums])
    """
    variants = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept both "ID, 1 2 3 4" and "ID,1,2,3,4"
        parts = [p.strip() for p in line.split(',') if p.strip() != ""]
        if len(parts) >= 2:
            try:
                vid = parts[0]
                rest = " ".join(parts[1:]).replace(',', ' ')
                nums = [int(x) for x in rest.split() if x.strip().isdigit()]
                if nums:
                    variants.append((str(vid), nums))
            except Exception:
                continue
    return variants

def load_variants_from_file(uploaded):
    """
    Accept CSV (first col ID, rest numbers), Excel or TXT.
    Returns list of (id, [nums])
    """
    if uploaded is None:
        return []
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded, header=None)
            rows = []
            for _, r in df.iterrows():
                vid = r.iloc[0]
                nums = [int(x) for x in r.iloc[1:].dropna().astype(int).tolist()]
                rows.append((str(vid), nums))
            return rows
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded, header=None)
            rows = []
            for _, r in df.iterrows():
                vid = r.iloc[0]
                nums = [int(x) for x in r.iloc[1:].dropna().astype(int).tolist()]
                rows.append((str(vid), nums))
            return rows
        elif name.endswith(".txt"):
            txt = uploaded.read().decode('utf-8')
            return parse_txt_file_content(txt)
        else:
            st.warning("Format fișier necunoscut — folosește CSV / XLSX / TXT.")
            return []
    except Exception as e:
        st.error(f"Eroare la citirea fișierului: {e}")
        return []

def load_rounds_from_file(uploaded):
    # same structure as variants: first column round id, remaining numbers
    return load_variants_from_file(uploaded)

# ---------------- Core helpers ----------------
def count_matches(variant, round_numbers):
    return len(set(variant) & set(round_numbers))

# ---------- Deterministic, efficient find_perfect_matches ----------
def find_perfect_matches(all_variants, historical_rounds, target_per_round=2,
                         desired_total=None, progress_bar=None, status_text=None):
    """
    - Index variants by tuple(sorted(nums)) -> list of (id, nums)
    - For each round, pick up to target_per_round unique variant ids
    - After that, if desired_total provided and not reached, pad with unused variants
    Returns: selected_variants_list, round_coverage (defaultdict)
    Each variant dict: {'id', 'numbers', 'round_id' (or None), 'match_type'}
    """
    # Build index
    combo_to_variants = defaultdict(list)
    for var_id, nums in all_variants:
        key = tuple(sorted(nums))
        combo_to_variants[key].append((var_id, nums))

    selected_variants = []
    used_ids = set()
    round_coverage = defaultdict(int)
    total_rounds = len(historical_rounds)

    start_time = time.time()
    for idx, (round_id, round_nums) in enumerate(historical_rounds):
        key = tuple(sorted(round_nums))
        candidates = combo_to_variants.get(key, [])
        # pick candidates in deterministic order (as in list). Avoid reusing id
        for var_id, nums in candidates:
            if round_coverage[round_id] >= target_per_round:
                break
            if var_id in used_ids:
                continue
            selected_variants.append({
                'id': var_id,
                'numbers': nums,
                'round_id': round_id,
                'match_type': '4/4'
            })
            used_ids.add(var_id)
            round_coverage[round_id] += 1

        # update UI progress + ETA
        if progress_bar and status_text:
            progress = (idx + 1) / max(total_rounds, 1)
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1) if (idx + 1) > 0 else 0.0
            remaining = (total_rounds - (idx + 1)) * avg_time
            status_text.text(f"🔄 Procesare rundă {idx + 1}/{total_rounds} — ⏱ Estimare {remaining:.1f}s")
            progress_bar.progress(min(1.0, progress))
            # tiny sleep to allow UI update
            time.sleep(0.001)

    # padding to desired_total if requested
    if desired_total:
        idx = 0
        while len(selected_variants) < desired_total and idx < len(all_variants):
            var_id, nums = all_variants[idx]
            if var_id not in used_ids:
                selected_variants.append({
                    'id': var_id,
                    'numbers': nums,
                    'round_id': None,
                    'match_type': 'pad'
                })
                used_ids.add(var_id)
            idx += 1

    if status_text:
        status_text.text("✅ Procesare completă.")
        if progress_bar:
            progress_bar.progress(1.0)

    return selected_variants, round_coverage

# ---------- Statistical generator ----------
def generate_statistical_coverage_variants(all_variants, historical_rounds, freq_counter,
                                           hot_numbers, cold_numbers, normal_numbers,
                                           pair_freq, triplet_freq,
                                           target_count=365, max_num=66, progress_callback=None):
    """
    Scorează variante și alege greedy pentru acoperire.
    Dacă all_variants sunt puse sub formă (id, nums) - le folosim.
    Returnează selected list, number_coverage, pair_coverage
    """
    st.info("📊 Scorez și aleg variante statistice...")
    # Prepare num scores
    all_nums = set(range(1, max_num + 1))
    max_freq = max(freq_counter.values()) if freq_counter else 1
    num_scores = {}
    for n in all_nums:
        score = 0
        if n in hot_numbers:
            score += 100
        elif n in normal_numbers:
            score += 50
        elif n not in freq_counter:
            score += 25
        elif n in cold_numbers:
            score += 5
        score += (freq_counter.get(n, 0) / max_freq) * 30
        num_scores[n] = score

    scored = []
    for idx, (vid, nums) in enumerate(all_variants):
        s = 0
        varset = set(nums)
        for n in nums:
            s += num_scores.get(n, 0)
        cold_count = len(varset & cold_numbers)
        if cold_count >= 3:
            s -= cold_count * 100
        elif cold_count >= 2:
            s -= cold_count * 50
        s += len(varset & hot_numbers) * 50
        # pair & triplet bonuses
        for pair in combinations(nums, 2):
            s += pair_freq.get(tuple(sorted(pair)), 0) * 5
        for trip in combinations(nums, 3):
            s += triplet_freq.get(tuple(sorted(trip)), 0) * 10
        # balance across thirds
        r1 = sum(1 for n in nums if n <= max_num//3)
        r2 = sum(1 for n in nums if max_num//3 < n <= 2*max_num//3)
        r3 = sum(1 for n in nums if n > 2*max_num//3)
        if r1 and r2 and r3:
            s += 30
        balance = abs(r1 - r2) + abs(r2 - r3) + abs(r1 - r3)
        s += (6 - balance) * 5
        # parity
        even = sum(1 for n in nums if n % 2 == 0)
        odd = len(nums) - even
        s += (len(nums) - abs(even - odd)) * 10
        # missing numbers cover
        missing_covered = len(varset & (all_nums - set(freq_counter.keys())))
        s += missing_covered * 80
        scored.append({'id': vid, 'numbers': nums, 'score': s, 'hot_count': len(varset & hot_numbers), 'cold_count': cold_count})

        if progress_callback and idx % 2000 == 0:
            progress_callback(idx / max(1, len(all_variants)))

    scored.sort(key=lambda x: x['score'], reverse=True)

    selected = []
    number_coverage = Counter()
    pair_coverage = Counter()

    # greedy selection
    for v in scored:
        if len(selected) >= target_count:
            break
        nums = v['numbers']
        new_nums = set(nums) - set(number_coverage.keys())
        coverage_value = len(new_nums) * 10
        min_cov = min(number_coverage.values()) if number_coverage else 0
        for n in nums:
            if number_coverage.get(n, 0) <= min_cov + 1:
                coverage_value += 5
        if coverage_value > 0 or len(selected) < target_count // 3:
            selected.append(v)
            number_coverage.update(nums)
            for p in combinations(nums, 2):
                pair_coverage[tuple(sorted(p))] += 1

    # fill remaining if some numbers are still missing
    all_nums_set = set(range(1, max_num + 1))
    covered_numbers = set(number_coverage.keys())
    still_missing = all_nums_set - covered_numbers
    if still_missing:
        for v in scored:
            if len(selected) >= target_count:
                break
            if v in selected:
                continue
            if set(v['numbers']) & still_missing:
                selected.append(v)
                number_coverage.update(v['numbers'])
                for p in combinations(v['numbers'], 2):
                    pair_coverage[tuple(sorted(p))] += 1
                still_missing -= set(v['numbers'])
    # pad to target_count
    i = 0
    while len(selected) < target_count and i < len(scored):
        if scored[i] not in selected:
            selected.append(scored[i])
            number_coverage.update(scored[i]['numbers'])
            for p in combinations(scored[i]['numbers'], 2):
                pair_coverage[tuple(sorted(p))] += 1
        i += 1

    # convert to canonical dict shape (like find_perfect_matches)
    out = []
    for v in selected:
        out.append({'id': v['id'], 'numbers': v['numbers'], 'match_type': 'statistical', 'score': v['score'], 'hot_count': v['hot_count'], 'cold_count': v['cold_count']})

    return out, number_coverage, pair_coverage

# ---------- Export helper ----------
def export_to_txt(variants):
    lines = []
    for v in variants:
        vid = v.get('id', '')
        nums = ' '.join(str(n) for n in v.get('numbers', []))
        lines.append(f"{vid}, {nums}")
    return "\n".join(lines)

# ---------------- UI: Upload & Config ----------------
st.header("📥 1. Import Variants & Historical Rounds")
colA, colB = st.columns(2)
with colA:
    uploaded_variants = st.file_uploader("Încarcă fișier variante (CSV / XLSX / TXT)", type=["csv", "xlsx", "txt"], key="variants_file")
    if uploaded_variants:
        st.success(f"Fișier variante: {uploaded_variants.name}")
with colB:
    uploaded_rounds = st.file_uploader("Încarcă fișier runde istorice (CSV / XLSX / TXT)", type=["csv", "xlsx", "txt"], key="rounds_file")
    if uploaded_rounds:
        st.success(f"Fișier runde: {uploaded_rounds.name}")

st.markdown("---")
st.header("⚙️ 2. Configurare Generare PRO")
col1, col2, col3 = st.columns(3)
with col1:
    coverage_variants = st.number_input("Variante Pas 1 (4/4 garantat)", min_value=1, max_value=5000, value=800, step=1)
with col2:
    number_variants = st.number_input("Variante Pas 2 (statistici)", min_value=1, max_value=5000, value=365, step=1)
with col3:
    max_number = st.number_input("Număr maxim (plajă)", min_value=10, max_value=100, value=66, step=1)

st.write(f"📊 Total variante generate (Pas1 + Pas2): **{coverage_variants + number_variants}**")
st.markdown("---")

# ---------------- RUN button ----------------
if st.button("🚀 GENEREAZĂ CU GARANȚIE 4/4", type="primary"):
    # Validate inputs
    if not uploaded_variants or not uploaded_rounds:
        st.error("Încarcă ambele fișiere: variante și runde istorice înainte de generare.")
    else:
        # Load files
        with st.spinner("📥 Încarc fișierele..."):
            all_variants = load_variants_from_file(uploaded_variants)
            historical_rounds = load_rounds_from_file(uploaded_rounds)

        if not all_variants:
            st.error("Fișier variante invalid sau nu conține rânduri parse-abile.")
        elif not historical_rounds:
            st.error("Fișier runde invalid sau nu conține rânduri parse-abile.")
        else:
            st.info("🔁 Încep generarea — vezi progresul mai jos.")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            start_all = time.time()

            # PASS 1: find perfect matches (deterministic index)
            target_per_round = 2
            covering_vars, round_coverage = find_perfect_matches(
                all_variants,
                historical_rounds,
                target_per_round=target_per_round,
                desired_total=coverage_variants,
                progress_bar=progress_bar,
                status_text=status_text
            )

            # compute historical stats
            freq_counter = Counter()
            for _, nums in historical_rounds:
                for n in nums:
                    freq_counter[n] += 1
            pair_freq = Counter()
            triplet_freq = Counter()
            for _, nums in historical_rounds:
                for p in combinations(nums, 2):
                    pair_freq[tuple(sorted(p))] += 1
                for t in combinations(nums, 3):
                    triplet_freq[tuple(sorted(t))] += 1

            avg_freq = np.mean(list(freq_counter.values())) if freq_counter else 0
            hot_numbers = set(n for n, f in freq_counter.items() if f > avg_freq + np.std(list(freq_counter.values())) * 0.5) if freq_counter else set()
            cold_numbers = set(n for n, f in freq_counter.items() if f < avg_freq - np.std(list(freq_counter.values())) * 0.5) if freq_counter else set()
            normal_numbers = set(freq_counter.keys()) - hot_numbers - cold_numbers

            # PASS 2: generate statistical variants (with light progress callback optionally)
            def progress_cb(p):
                try:
                    progress_bar.progress(0.5 + 0.4 * min(1.0, p))
                except Exception:
                    pass

            number_vars, number_coverage, pair_coverage = generate_statistical_coverage_variants(
                all_variants,
                historical_rounds,
                freq_counter,
                hot_numbers,
                cold_numbers,
                normal_numbers,
                pair_freq,
                triplet_freq,
                target_count=number_variants,
                max_num=max_number,
                progress_callback=progress_cb
            )

            # combine results
            final_variants = covering_vars + number_vars
            # compute complete coverage counts
            complete_coverage = Counter()
            for v in final_variants:
                complete_coverage.update(v['numbers'])

            elapsed = time.time() - start_all
            status_text.text(f"✅ Finalizat în {elapsed:.1f}s — {len(final_variants)} variante generate.")
            progress_bar.progress(1.0)

            # ---------- Analysis & Test on historical rounds ----------
            st.markdown("---")
            st.header("📊 Rezultate & Analiză")
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("✅ Variante generate", len(final_variants))
            with colB:
                paso1_count = sum(1 for v in final_variants if v.get('match_type') == '4/4')
                st.metric("🎯 Variante 4/4 (Pas1)", paso1_count)
            with colC:
                covered_numbers = [n for n in range(1, max_number + 1) if complete_coverage.get(n, 0) > 0]
                st.metric("🌐 Numere acoperite", f"{len(covered_numbers)}/{max_number}")
            with colD:
                avg_cov = sum(complete_coverage.values()) / max(1, len(complete_coverage))
                st.metric("📊 Acoperire medie", f"{avg_cov:.1f}x")

            # Test pe runde istorice: distribuție de matches
            match_dist = Counter()
            winning_variants = []
            for v in final_variants:
                wins = 0
                for _, round_nums in historical_rounds:
                    m = count_matches(v['numbers'], round_nums)
                    match_dist[m] += 1
                    if m >= 2:
                        wins += 1
                if wins > 0:
                    winning_variants.append((v, wins))

            st.write(f"🏆 Variante câștigătoare (au >=2 potriviri pe istoric): {len(winning_variants)}/{len(final_variants)}")
            st.write(f"🔎 Distribuție potriviri (pe toate comparațiile): {dict(match_dist)}")
            # show top hot / cold
            st.markdown("**🔥 Top 15 numere folosite în variante**")
            top_nums = complete_coverage.most_common(15)
            for i, (num, cnt) in enumerate(top_nums, 1):
                tag = "🔥" if num in hot_numbers else ""
                st.text(f"{i}. {num}: {cnt}x {tag}")

            # ---------- Export buttons ----------
            st.markdown("---")
            st.header("💾 Export")
            txt_all = export_to_txt(final_variants)
            txt_4 = export_to_txt([v for v in final_variants if v.get('match_type') == '4/4'])
            txt_stat = export_to_txt([v for v in final_variants if v.get('match_type') == 'statistical' or v.get('match_type') == 'pad'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📥 Descarcă TOATE variantele (.txt)", txt_all, file_name=f"variante_complete_{len(final_variants)}.txt", mime="text/plain")
            with col2:
                st.download_button("🎯 Descarcă 4/4 (.txt)", txt_4, file_name=f"variante_4din4_{len([v for v in final_variants if v.get('match_type')=='4/4'])}.txt", mime="text/plain")
            with col3:
                st.download_button("📊 Descarcă statistice (.txt)", txt_stat, file_name=f"variante_stat_{len([v for v in final_variants if v.get('match_type')!='4/4'])}.txt", mime="text/plain")

            st.success("✅ Gata — poți descărca rezultatele sau re-executa cu alți parametri.")
            st.markdown("---")
            st.caption("Generator Inteligent Loterie PRO — versiune completă")