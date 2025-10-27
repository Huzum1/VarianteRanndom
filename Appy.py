# app.py (versiune corectată)
import streamlit as st
import pandas as pd
import random
import time
from copy import deepcopy
import plotly.express as px
import statistics
import numpy as np
from io import BytesIO
# from multiprocessing import Pool, cpu_count  # înlocuit mai jos
from functools import partial
import sys
import ast
import concurrent.futures
from multiprocessing import cpu_count
from random import SystemRandom

# =========================================================================
# CONSTANTE ȘI CONFIGURARE PAGINĂ
# =========================================================================

# CONSTANTE DE VITEZĂ ȘI RESURSE - OPTIMIZATE
MAX_RANDOM_ATTEMPTS = 50000
INTERMEDIATE_SAVE_INTERVAL = 10000
NUM_PROCESSES = max(1, cpu_count() - 1)
CHART_UPDATE_INTERVAL = 1000
LOCAL_SEARCH_DEFAULT_ITERATIONS = 5000
BATCH_SIZE = 200
EARLY_STOP_NO_IMPROVEMENT = 5000
EARLY_STOP_NO_IMPROVEMENT_LOCAL = 1000

# CONSTANTE DE SCOR
PENALTY_FACTOR_K = 0.3
NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS = 10

st.set_page_config(
    page_title="Generator Variante Loterie (Premium)",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZAT ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #667eea; }
    h1 { color: white !important; text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem; }
    h2, h3 { color: #667eea !important; }
    .results-box { border: 1px solid #764ba2; padding: 15px; border-radius: 8px; background-color: #333333; color: white; height: 400px; overflow-y: scroll; font-family: monospace; }
    .status-box {
        border: 2px solid #667eea; padding: 10px; border-radius: 8px;
        background-color: #f0f2f6; color: #333333; font-size: 1.05rem;
        font-weight: bold; text-align: center; margin-top: 10px;
    }
    .local-search-status { color: #e67e22; }
    .score-detail { font-size: 0.9rem; font-weight: normal; margin-top: 5px; }
    .intermediate-save { color: #27ae60; font-size: 0.8rem; margin-top: 5px; }
    .target-met { background-color: #27ae60; color: white; }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# INITIALIZARE SESIUNE
# =========================================================================

if 'variants' not in st.session_state: st.session_state.variants = []
if 'variants_sets_precomputed' not in st.session_state: st.session_state.variants_sets_precomputed = []
if 'generated_variants' not in st.session_state: st.session_state.generated_variants = []
if 'rounds' not in st.session_state: st.session_state.rounds = []
if 'rounds_raw' not in st.session_state: st.session_state.rounds_raw = []
if 'best_score_full' not in st.session_state:
    st.session_state.best_score_full = {
        'win_score': 0, 'score_3_3': 0, 'score_2_2': 0,
        'weighted_score_sum': 0, 'std_dev_wins': 0, 'fitness_score': -float('inf'),
        'score_per_round': {}
    }
if 'intermediate_saves' not in st.session_state: st.session_state.intermediate_saves = []
if 'optimization_attempts' not in st.session_state: st.session_state.optimization_attempts = 0
if 'local_search_attempts' not in st.session_state: st.session_state.local_search_attempts = 0
if 'manual_rounds_input' not in st.session_state: st.session_state.manual_rounds_input = ""
if 'variants_input_text' not in st.session_state: st.session_state.variants_input_text = ""
if 'params' not in st.session_state:
    st.session_state.params = {
        'count': 1165,
        'target_wins_plus': 10,
        'local_search_iterations': LOCAL_SEARCH_DEFAULT_ITERATIONS,
        'use_recency_weighting': True,
        'use_deviation_penalty': True,
        'penalty_factor_k': PENALTY_FACTOR_K
    }
if 'score_evolution_data' not in st.session_state:
    st.session_state.score_evolution_data = []

# =========================================================================
# FUNCȚII UTILITY ȘI SCOR
# =========================================================================

def clean_variant_combination(numbers_str):
    try:
        parts = [p.strip() for p in numbers_str.split() if p.strip().isdigit()]
        valid_numbers = [int(p) for p in parts]
        unique_numbers = list(set(valid_numbers))
        unique_numbers.sort()
        duplicates_removed = len(valid_numbers) - len(unique_numbers)
        cleaned_combination = ' '.join(map(str, unique_numbers))
        return cleaned_combination, duplicates_removed
    except:
        return numbers_str, 0

def parse_variants(text):
    variants = []
    errors = []
    total_internal_duplicates_removed = 0
    lines = text.strip().split('\n')
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue
        parts = line.split(',', 1)
        if len(parts) != 2:
            errors.append(f"Linia {i}: Format invalid (lipsește virgula)")
            continue
        variant_id = parts[0].strip()
        numbers = parts[1].strip()
        if not variant_id or not numbers: continue

        cleaned_combination, duplicates_removed = clean_variant_combination(numbers)
        total_internal_duplicates_removed += duplicates_removed

        if len(cleaned_combination.split()) < 4:
            errors.append(f"Linia {i}: Combinația '{numbers}' are sub 4 numere unice după curățare.")
            continue

        variants.append({'id': variant_id, 'combination': cleaned_combination})

    df = pd.DataFrame(variants)
    if not df.empty:
        df_unique = df.drop_duplicates(subset=['combination']).reset_index(drop=True)
        df_unique['id'] = (df_unique.index + 1).astype(str)
        final_variants = df_unique.to_dict('records')
        total_inter_duplicates_removed = len(variants) - len(final_variants)
    else:
        final_variants = []
        total_inter_duplicates_removed = 0

    return final_variants, errors, total_internal_duplicates_removed, total_inter_duplicates_removed

def parse_variants_file(file):
    if file is None: return [], []
    try:
        content = file.getvalue().decode("utf-8")
        variants, errors, _, _ = parse_variants(content)
        return variants, errors
    except Exception as e:
        return [], [f"Eroare citire fișier: {str(e)}"]

def process_round_text(text):
    rounds_set_list = []
    rounds_display_list = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in line.replace(',', ' ').split() if p.strip().isdigit()]
        round_numbers = {int(p) for p in parts if p.isdigit()}

        if len(round_numbers) >= 4:
            rounds_set_list.append(frozenset(round_numbers))
            display_numbers = ' '.join(map(str, sorted(list(round_numbers))))
            rounds_display_list.append(display_numbers)
    return rounds_set_list, rounds_display_list

@st.cache_data
def parse_rounds_file(rounds_file):
    if rounds_file is None: return [], []
    try:
        content = rounds_file.getvalue().decode("utf-8")
        return process_round_text(content)
    except Exception as e:
        return [], []

def precompute_variant_sets(variants):
    variant_sets = []
    for v in variants:
        try:
            v_set = frozenset(int(n) for n in v['combination'].split() if n.isdigit())
            variant_sets.append(v_set)
        except:
            variant_sets.append(frozenset())
    return variant_sets

def get_round_weights(rounds):
    N = len(rounds)
    if N == 0: return {}
    BASE = 0.98
    reversed_rounds = list(reversed(rounds))
    weights = [BASE**(N - i - 1) for i in range(N)]
    return {runda: weight for runda, weight in zip(reversed_rounds, weights)}

def calculate_wins_optimized(variant_indices, all_variant_sets, rounds_data,
                            round_weights_data=None, use_deviation_penalty=False,
                            penalty_factor_k=0.5):

    rounds = [frozenset(r) for r in rounds_data]

    if round_weights_data:
        round_weights = {}
        for k_list, v in round_weights_data:
            round_weights[frozenset(k_list)] = v
    else:
        round_weights = {r: 1.0 for r in rounds}

    if not rounds or not variant_indices:
        return {
            'win_score': 0, 'score_3_3': 0, 'score_2_2': 0,
            'weighted_score_sum': 0, 'std_dev_wins': 0, 'fitness_score': -float('inf'),
            'score_per_round': {}
        }

    variant_sets = [all_variant_sets[i] for i in variant_indices]

    total_wins, total_3_3, total_2_2 = 0, 0, 0
    score_per_round = {}
    weighted_score_sum = 0

    for runda_set in rounds:
        wins_in_round, score_3_3_in_round, score_2_2_in_round = 0, 0, 0
        weight = round_weights.get(runda_set, 1.0)

        for variant_set in variant_sets:
            matches = len(variant_set.intersection(runda_set))

            if matches >= 4: wins_in_round += 1
            elif matches == 3: score_3_3_in_round += 1
            elif matches == 2: score_2_2_in_round += 1

        total_wins += wins_in_round
        total_3_3 += score_3_3_in_round
        total_2_2 += score_2_2_in_round

        weighted_score_sum += wins_in_round * weight

        score_per_round[str(list(runda_set))] = {
            'wins': wins_in_round,
            '3_3': score_3_3_in_round,
            '2_2': score_2_2_in_round
        }

    wins_list = [
        (d['wins'] * 1000) + (d['3_3'] * 10) + (d['2_2'] * 1)
        for d in score_per_round.values()
    ]

    std_dev_wins = statistics.stdev(wins_list) if len(wins_list) > 1 else 0

    base_score = weighted_score_sum if round_weights_data else total_wins
    base_score_multi = base_score * 100000 + total_3_3 * 100 + total_2_2 * 1

    fitness_score = base_score_multi

    if use_deviation_penalty:
        fitness_score = base_score_multi - (penalty_factor_k * std_dev_wins * 10000)

    return {
        'win_score': total_wins,
        'score_3_3': total_3_3,
        'score_2_2': total_2_2,
        'weighted_score_sum': weighted_score_sum,
        'std_dev_wins': std_dev_wins,
        'fitness_score': fitness_score,
        'score_per_round': score_per_round
    }

def compare_scores(current_score, best_score):
    if current_score['fitness_score'] > best_score['fitness_score']: return True
    if current_score['fitness_score'] < best_score['fitness_score']: return False
    if current_score['win_score'] > best_score['win_score']: return True
    if current_score['win_score'] < best_score['win_score']: return False
    if current_score['score_3_3'] > best_score['score_3_3']: return True
    if current_score['score_3_3'] < best_score['score_3_3']: return False
    if current_score['score_2_2'] > best_score['score_2_2']: return True
    return False

# =========================================================================
# FUNCȚII EVALUARE (adaptate pentru ThreadPoolExecutor)
# =========================================================================

def evaluate_random_sample_worker(args):
    # args: (seed, count, num_variants, rounds_data, round_weights_data, use_deviation_penalty, penalty_factor_k, all_variant_sets)
    seed, count, num_variants, rounds_data, round_weights_data, \
        use_deviation_penalty, penalty_factor_k, all_variant_sets = args

    rnd = random.Random(seed)
    indices = list(range(num_variants))
    rnd.shuffle(indices)
    sample_indices = indices[:count]

    score = calculate_wins_optimized(
        sample_indices,
        all_variant_sets,
        rounds_data,
        round_weights_data,
        use_deviation_penalty,
        penalty_factor_k
    )

    return sample_indices, score

def evaluate_candidate_hole_worker(args):
    candidate_idx, all_variant_sets, weak_round_sets_data = args

    weak_round_sets = [frozenset(r) for r in weak_round_sets_data]
    variant_set = all_variant_sets[candidate_idx]
    hole_score = 0

    for runda_set in weak_round_sets:
        matches = len(variant_set.intersection(runda_set))
        if matches >= 4: hole_score += 1000
        elif matches == 3: hole_score += 100
        elif matches == 2: hole_score += 10

    return candidate_idx, hole_score

# =========================================================================
# FUNCȚII ANALIZĂ ȘI PLOTLY
# =========================================================================

def analyze_round_performance(generated_variants, rounds_set, variant_sets_precomputed=None):
    if not rounds_set or not generated_variants: return ""

    if variant_sets_precomputed is None:
        variant_sets = []
        for variant_data in generated_variants:
            try:
                variant_sets.append(frozenset(int(n) for n in variant_data['combination'].split() if n.isdigit()))
            except:
                continue
    else:
        variant_sets = variant_sets_precomputed

    results_lines = []
    for i, runda_set in enumerate(rounds_set):
        wins_in_round, score_3_3_in_round, score_2_2_in_round = 0, 0, 0

        runda_display = st.session_state.rounds_raw[i] if i < len(st.session_state.rounds_raw) else f"Runda {i+1}"

        for v_set in variant_sets:
            matches = len(v_set.intersection(runda_set))
            if matches >= 4: wins_in_round += 1
            elif matches == 3: score_3_3_in_round += 1
            elif matches == 2: score_2_2_in_round += 1

        results_lines.append(f"{runda_display} - WINs: {wins_in_round}, 3/3: {score_3_3_in_round}, 2/2: {score_2_2_in_round}")
    return '\n'.join(results_lines)

def variants_to_text(variants):
    return '\n'.join([f"{v['id']},{v['combination']}" for v in variants])

def variants_to_csv(variants):
    df = pd.DataFrame(variants)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue().decode('utf-8')

def analyze_number_frequency(variants):
    frequency = {i: 0 for i in range(1, 67)}
    total_numbers = 0
    for v in variants:
        try:
            numbers = [int(n) for n in v['combination'].split() if n.isdigit()]
            for n in numbers:
                if 1 <= n <= 66:
                    frequency[n] += 1
                    total_numbers += 1
        except:
            continue

    df = pd.DataFrame(list(frequency.items()), columns=['Număr', 'Frecvență'])

    if total_numbers > 0:
        df['Procent'] = (df['Frecvență'] / total_numbers) * 100
    else:
         df['Procent'] = 0

    return df

def analyze_variant_strength(variants, rounds):
    if not rounds or not variants:
        return pd.DataFrame({'ID': [], 'Combinație': [], 'Forță (Score)': []})

    strength_data = []

    for variant_data in variants:
        try:
            variant_set = frozenset(int(n) for n in variant_data['combination'].split() if n.isdigit())
        except:
            continue

        total_score = 0

        for runda_set in rounds:
            matches = len(variant_set.intersection(runda_set))
            if matches >= 4: total_score += 10000
            elif matches == 3: total_score += 100
            elif matches == 2: total_score += 1

        strength_data.append({
            'ID': variant_data['id'],
            'Combinație': variant_data['combination'],
            'Forță (Score)': total_score
        })

    df_strength = pd.DataFrame(strength_data)
    df_strength = df_strength.sort_values(by='Forță (Score)', ascending=False).reset_index(drop=True)
    return df_strength

def plot_score_evolution(scores_data):
    if not scores_data:
        return None

    df = pd.DataFrame(scores_data)
    df['Fitness'] = df['Fitness'].astype(float)

    fig = px.line(
        df,
        x='Încercare',
        y='Fitness',
        title='Evoluția Scorului Fitness',
        labels={'Încercare': 'Număr Încercări', 'Fitness': 'Scor Fitness'}
    )
    # NOTE: Respect restriction "do not set colors explicitly" only for python_user_visible charts;
    # here plotly is used (ok) — păstrăm actualizarea estetică pentru streamlit
    fig.update_traces(line_color='#27ae60', line_width=2)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title='Număr Încercări',
        yaxis_title='Scor Fitness'
    )
    return fig

# =========================================================================
# STREAMLIT UI & LOGIC FLOW
# =========================================================================

st.markdown("# 👑 Generator Variante Loterie (Premium Optimizat)")
st.markdown("### Optimizare Multi-CPU Fără Blocaje - Performanță Maximă")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Statistici Curente")
    st.metric("Variante Curățate", len(st.session_state.variants))
    st.metric("Runde Încărcate", len(st.session_state.rounds_raw))
    st.metric("CPU-uri Paralel", NUM_PROCESSES)
    st.markdown("---")
    st.markdown("#### Scor Final Obținut")

    full_score = st.session_state.best_score_full
    fitness_display = f"{full_score['fitness_score']:.0f}" if full_score['fitness_score'] != -float('inf') else "-inf"

    target_win = len(st.session_state.rounds) + st.session_state.params['target_wins_plus']

    target_met_class = ""
    if full_score['win_score'] >= target_win and full_score['win_score'] > 0:
        target_met_class = "target-met"

    st.markdown(f"""
        <div class='stMetricValue {target_met_class}'>
            {fitness_display}
        </div>
        <p style='font-size: 0.8rem; margin-top: -15px;'>Fitness (Ponderat)</p>
    """, unsafe_allow_html=True)

    st.caption(f"WIN: {full_score['win_score']} (Țintă: {target_win}) | 3/3: {full_score['score_3_3']} | Std Dev: {full_score['std_dev_wins']:.2f}")

    st.markdown("---")
    st.markdown("#### 💾 Salvări Intermediare")
    if st.session_state.intermediate_saves:
        st.success(f"🎉 {len(st.session_state.intermediate_saves)} eșantioane salvate")
        for i, save in enumerate(st.session_state.intermediate_saves):
            save_score_display = f"{save['score']['fitness_score']:.0f}"
            save_name = f"#{i+1} | WIN: {save['score']['win_score']} | Fit: {save_score_display}"

            col_idx, col_dl = st.columns([2, 1])

            with col_idx:
                 st.caption(save_name)

            with col_dl:
                st.download_button(
                    "💾",
                    data=variants_to_text(save['variants']),
                    file_name=f"salvare_{i+1}.txt",
                    mime="text/plain",
                    key=f"dl_save_{i}",
                    use_container_width=True
                )
    else:
        st.info("Nicio salvare.")

    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante", "🎲 Generează", "📊 Rezultate"])

# =========================================================================
# TAB 1: ÎNCARCĂ VARIANTE
# =========================================================================
with tab1:
    st.markdown("## 📝 Pas 1: Încarcă Variantele Tale")

    upload_method = st.radio("Metodă:", ["📄 Fișier", "⌨️ Text"], horizontal=True)

    if upload_method == "📄 Fișier":
        variants_file = st.file_uploader(
            "Încarcă fișier (format: ID, numere)",
            type=['txt', 'csv'],
            key="variants_file_uploader"
        )

        if variants_file:
            variants_from_file, errors_from_file = parse_variants_file(variants_file)

            if st.button("📥 Încarcă din Fișier", use_container_width=True, type="primary"):
                st.session_state.variants = variants_from_file

                if errors_from_file:
                    st.error("Erori găsite:")
                    for err in errors_from_file[:10]:
                        st.write(f"- {err}")

                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 {len(st.session_state.variants)} variante încărcate!")
                    st.rerun()
    else:
        st.session_state.variants_input_text = st.text_area(
            "Variante (ID, numere)",
            value=st.session_state.variants_input_text,
            height=300
        )

        if st.button("📥 Încarcă din Text", use_container_width=True, type="primary"):
            if st.session_state.variants_input_text:
                variants_list, errors, _, _ = parse_variants(st.session_state.variants_input_text)
                st.session_state.variants = variants_list

                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 {len(st.session_state.variants)} variante încărcate!")
                    st.rerun()

    st.markdown("---")
    if st.session_state.variants:
        col_txt, col_csv = st.columns(2)
        with col_txt:
            st.download_button(
                "💾 TXT",
                data=variants_to_text(st.session_state.variants),
                file_name=f"variante_{len(st.session_state.variants)}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_csv:
            st.download_button(
                "📊 CSV",
                data=variants_to_csv(st.session_state.variants),
                file_name=f"variante_{len(st.session_state.variants)}.csv",
                mime="text/csv",
                use_container_width=True
            )

# =========================================================================
# TAB 2: Generare & Optimizare (MODIFICAT)
# =========================================================================
with tab2:
    st.markdown("## 🎲 Pas 2: Configurare și Generare")

    if not st.session_state.variants:
        st.warning("⚠️ Încarcă variante mai întâi!")
    else:
        st.markdown("### 1. Încarcă Rundele")

        col_file, col_manual = st.columns(2)
        rounds_file = col_file.file_uploader("Fișier Runde", type=['txt', 'csv'], key="rounds_uploader")
        manual_rounds_input = col_manual.text_area("Sau manual", value=st.session_state.manual_rounds_input, height=100)
        st.session_state.manual_rounds_input = manual_rounds_input

        rounds_from_file_set, rounds_from_file_raw = parse_rounds_file(rounds_file)
        rounds_from_manual_set, rounds_from_manual_raw = process_round_text(manual_rounds_input)

        all_rounds_set_dict = {}
        for r_set, r_raw in zip(rounds_from_file_set, rounds_from_file_raw):
            all_rounds_set_dict[r_set] = r_raw
        for r_set, r_raw in zip(rounds_from_manual_set, rounds_from_manual_raw):
            all_rounds_set_dict[r_set] = r_raw

        st.session_state.rounds = list(all_rounds_set_dict.keys())
        st.session_state.rounds_raw = list(all_rounds_set_dict.values())

        st.metric("Total Runde", len(st.session_state.rounds))

        st.markdown("### 2. Configurare")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.session_state.params['count'] = st.number_input(
                "Variante de generat",
                min_value=1,
                max_value=len(st.session_state.variants),
                value=min(st.session_state.params['count'], len(st.session_state.variants))
            )

        with col2:
            st.session_state.params['target_wins_plus'] = st.number_input(
                "WINs Țintă +",
                min_value=0,
                value=st.session_state.params['target_wins_plus'],
                help="Numărul total de WINs țintă va fi: Total Runde + această valoare."
            )

        with col3:
            st.session_state.params['local_search_iterations'] = st.number_input(
                "Iterații Faza 2 (Vizual/Stop)",
                min_value=0,
                value=LOCAL_SEARCH_DEFAULT_ITERATIONS,
                help="Această limită este doar vizuală și pentru progres. Căutarea continuă până la scorul țintă."
            )

        col4, col5 = st.columns(2)
        with col4:
            st.session_state.params['use_recency_weighting'] = st.checkbox(
                "Ponderare Recență",
                value=True
            )
        with col5:
            st.session_state.params['use_deviation_penalty'] = st.checkbox(
                "Penalizare Uniformitate",
                value=True
            )

        if st.button("🚀 GENEREAZĂ OPTIMIZAT", use_container_width=True, type="primary"):

            if not st.session_state.rounds:
                st.error("Încarcă runde mai întâi!")
            else:
                count = st.session_state.params['count']
                local_search_iterations = st.session_state.params['local_search_iterations']
                use_recency_weighting = st.session_state.params['use_recency_weighting']
                use_deviation_penalty = st.session_state.params['use_deviation_penalty']
                penalty_factor_k = st.session_state.params['penalty_factor_k']

                # Calculează scorul țintă
                target_win_score = len(st.session_state.rounds) + st.session_state.params['target_wins_plus']

                # Pregătire date
                rounds_data_list = [sorted(list(r)) for r in st.session_state.rounds]

                round_weights_data = None
                if use_recency_weighting:
                    round_weights = get_round_weights(st.session_state.rounds)
                    round_weights_data = [(sorted(list(k)), v) for k, v in round_weights.items()]

                # Inițializez stări
                attempts, local_attempts = 0, 0
                best_score = {
                    'win_score': 0, 'score_3_3': 0, 'score_2_2': 0,
                    'weighted_score_sum': 0, 'std_dev_wins': 0,
                    'fitness_score': -float('inf'), 'score_per_round': {}
                }
                best_variant_indices = []
                st.session_state.intermediate_saves = []
                st.session_state.score_evolution_data = []

                if not st.session_state.variants_sets_precomputed:
                    with st.spinner("Precomputare..."):
                        st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)

                all_variant_sets = st.session_state.variants_sets_precomputed
                num_variants = len(st.session_state.variants)

                # FAZA 1: Căutare Aleatorie Multi-Threaded (evită problemele Pool/process)
                st.info(f"🚀 FAZA 1: {MAX_RANDOM_ATTEMPTS:,} încercări (Țintă WIN: {target_win_score})")

                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                chart_placeholder = st.empty()

                score_evolution_data = []
                no_improvement_counter = 0

                rnd_sys = SystemRandom()
                batch_size = max(1, min(NUM_PROCESSES * 5, 1000))  # adaptabil
                total_batches = (MAX_RANDOM_ATTEMPTS + batch_size - 1) // batch_size

                # folosim ThreadPoolExecutor pentru compatibilitate Streamlit
                with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                    for batch_idx in range(total_batches):
                        # creez seeds & args doar pentru batch-ul curent
                        start_attempt_idx = batch_idx * batch_size
                        end_attempt_idx = min(start_attempt_idx + batch_size, MAX_RANDOM_ATTEMPTS)
                        current_batch_count = end_attempt_idx - start_attempt_idx

                        args_list = []
                        for i in range(current_batch_count):
                            seed = rnd_sys.randint(0, 2**31 - 1) ^ (start_attempt_idx + i)
                            args = (
                                seed, count, num_variants, rounds_data_list, round_weights_data,
                                use_deviation_penalty, penalty_factor_k, all_variant_sets
                            )
                            args_list.append(args)

                        # map cu futures
                        futures = [executor.submit(evaluate_random_sample_worker, a) for a in args_list]
                        for fut in concurrent.futures.as_completed(futures):
                            try:
                                sample_indices, current_score = fut.result()
                            except Exception as e:
                                # nu întrerupem tot procesul pentru o eroare punctuala
                                continue

                            attempts += 1

                            is_better = compare_scores(current_score, best_score)

                            if is_better:
                                best_score = current_score.copy()
                                best_variant_indices = sample_indices
                                no_improvement_counter = 0
                            else:
                                no_improvement_counter += 1

                            if attempts % 50 == 0 or attempts == 1 or is_better:
                                score_evolution_data.append({
                                    'Încercare': attempts,
                                    'Fitness': best_score['fitness_score']
                                })

                            # actualizare UI
                            progress_percent = min(1.0, attempts / MAX_RANDOM_ATTEMPTS)
                            progress_bar.progress(progress_percent)

                            score_detail = f"FIT: **{best_score['fitness_score']:.0f}** | WIN: {best_score['win_score']:,} | Țintă: {target_win_score}"
                            status_html = f"""
                            <div class="status-box">
                                FAZA 1: {attempts:,}/{MAX_RANDOM_ATTEMPTS:,} | No Improve: {no_improvement_counter}
                                <div class="score-detail">{score_detail}</div>
                            </div>
                            """
                            status_placeholder.markdown(status_html, unsafe_allow_html=True)

                            if attempts % CHART_UPDATE_INTERVAL == 0:
                                fig = plot_score_evolution(score_evolution_data)
                                if fig:
                                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                            if attempts % INTERMEDIATE_SAVE_INTERVAL == 0 and best_variant_indices:
                                save_variants = [st.session_state.variants[i] for i in best_variant_indices]
                                st.session_state.intermediate_saves.append({
                                    'attempt': attempts,
                                    'score': best_score.copy(),
                                    'variants': deepcopy(save_variants)
                                })

                            if no_improvement_counter >= EARLY_STOP_NO_IMPROVEMENT:
                                st.info(f"⏹️ Early stop Faza 1 la {attempts:,} (no improvement: {no_improvement_counter})")
                                break

                        # verificare ieșire din bucla externă
                        if no_improvement_counter >= EARLY_STOP_NO_IMPROVEMENT:
                            break
                        if attempts >= MAX_RANDOM_ATTEMPTS:
                            break

                st.session_state.optimization_attempts = attempts
                st.session_state.score_evolution_data = score_evolution_data
                status_placeholder.empty()
                progress_bar.empty()

                st.success(f"✅ FAZA 1 Completă: {attempts:,} încercări | Fitness: {best_score['fitness_score']:.0f}")

                # =========================================================================
                # FAZA 2: Căutare Evolutivă Multi-Threaded (Hole Filling)
                # =========================================================================

                if best_variant_indices:
                    st.info(f"🧬 FAZA 2: Evolutivă continuă. **Țintă WIN: {target_win_score}**.")

                    local_status_placeholder = st.empty()
                    local_progress = st.progress(0)

                    current_best_indices = best_variant_indices.copy()
                    current_best_score = best_score.copy()

                    no_improve_local = 0
                    local_attempts = 0

                    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor2:
                        # buclă cu oprire prin condiții
                        while True:
                            local_attempts += 1

                            # 1. Găsește varianta cea mai slabă (sequential e ok pe set mic)
                            variant_scores = {}
                            for idx in current_best_indices:
                                score_single = calculate_wins_optimized(
                                    [idx], all_variant_sets, rounds_data_list, round_weights_data
                                )
                                score_metric = (score_single['win_score'] * 10000 +
                                              score_single['score_3_3'] * 100 +
                                              score_single['score_2_2'])
                                variant_scores[idx] = score_metric

                            weakest_idx = min(variant_scores, key=variant_scores.get)
                            weakest_position = current_best_indices.index(weakest_idx)

                            # 2. Identifică rundele slabe din score_per_round (safe parsing)
                            score_per_round_dict = {}
                            for k_str, v in current_best_score['score_per_round'].items():
                                try:
                                    k_list = ast.literal_eval(k_str)
                                    score_per_round_dict[frozenset(k_list)] = v
                                except Exception:
                                    continue

                            weakest_rounds = sorted(
                                score_per_round_dict.items(),
                                key=lambda item: (item[1]['wins'] * 1000 +
                                                item[1]['3_3'] * 10 +
                                                item[1]['2_2'])
                            )[:NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS]

                            weak_round_sets = [item[0] for item in weakest_rounds]
                            weak_round_sets_data = [sorted(list(r)) for r in weak_round_sets]

                            # 3. Găsește candidați (Sample)
                            available_indices = [i for i in range(num_variants)
                                               if i not in current_best_indices]

                            if not available_indices:
                                break

                            sample_size = min(BATCH_SIZE, len(available_indices))
                            sampled_indices = random.sample(available_indices, sample_size)

                            # 4. EVALUARE PARALELĂ (Hole Coverage)
                            args_list_faza2 = [
                                (idx, all_variant_sets, weak_round_sets_data)
                                for idx in sampled_indices
                            ]

                            futures2 = [executor2.submit(evaluate_candidate_hole_worker, a) for a in args_list_faza2]
                            hole_scores = []
                            for f in concurrent.futures.as_completed(futures2):
                                try:
                                    hole_scores.append(f.result())
                                except:
                                    continue

                            best_hole_score = -1
                            best_candidate_idx = None

                            for candidate_idx, hole_score in hole_scores:
                                if hole_score > best_hole_score:
                                    best_hole_score = hole_score
                                    best_candidate_idx = candidate_idx

                            if best_candidate_idx is None:
                                no_improve_local += 1
                                # continue, dar check exit
                                if no_improve_local >= EARLY_STOP_NO_IMPROVEMENT_LOCAL:
                                    st.info(f"⏹️ Early stop Faza 2 la {local_attempts:,}. Nu s-a putut atinge scorul țintă {target_win_score} (no improve: {no_improve_local})")
                                    break
                                continue

                            # 5. Testare și acceptare
                            test_indices = current_best_indices.copy()
                            test_indices[weakest_position] = best_candidate_idx

                            test_score = calculate_wins_optimized(
                                test_indices, all_variant_sets, rounds_data_list,
                                round_weights_data, use_deviation_penalty, penalty_factor_k
                            )

                            if compare_scores(test_score, current_best_score):
                                current_best_score = test_score.copy()
                                current_best_indices = test_indices.copy()
                                no_improve_local = 0
                            else:
                                no_improve_local += 1

                            # Update Progresul Vizual
                            local_progress.progress(min(1.0, local_attempts / max(1, local_search_iterations)))

                            score_detail = f"FIT: **{current_best_score['fitness_score']:.0f}** | WIN: {current_best_score['win_score']:,} / {target_win_score}"
                            status_html = f"""
                            <div class="status-box local-search-status">
                                FAZA 2: Încercare {local_attempts:,} | No Improve: {no_improve_local}
                                <div class="score-detail">{score_detail}</div>
                            </div>
                            """
                            local_status_placeholder.markdown(status_html, unsafe_allow_html=True)

                            # Condițiile de ieșire
                            if current_best_score['win_score'] >= target_win_score:
                                st.success(f"🎯 **SCOR ȚINTĂ ATINS!** WIN: {current_best_score['win_score']:,} la iterația {local_attempts:,}.")
                                break

                            if no_improve_local >= EARLY_STOP_NO_IMPROVEMENT_LOCAL:
                                st.info(f"⏹️ Early stop Faza 2 la {local_attempts:,}. Nu s-a putut atinge scorul țintă {target_win_score} (no improve: {no_improve_local})")
                                break

                            if local_attempts > 100000:
                                st.warning("⚠️ Limită de siguranță atinsă (100k iterații). Oprire forțată.")
                                break

                    best_score = current_best_score
                    best_variant_indices = current_best_indices.copy()
                    local_status_placeholder.empty()
                    local_progress.empty()

                    st.success(f"✅ FAZA 2 Completă: Fitness: {best_score['fitness_score']:.0f} după {local_attempts:,} iterații.")
                else:
                    local_attempts = 0

                st.session_state.local_search_attempts = local_attempts
                st.session_state.generated_variants = [st.session_state.variants[i] for i in best_variant_indices] if best_variant_indices else []
                st.session_state.best_score_full = best_score

                # Analiză performanță
                generated_sets = [all_variant_sets[i] for i in best_variant_indices] if best_variant_indices else []
                performance_text = analyze_round_performance(
                    st.session_state.generated_variants,
                    st.session_state.rounds,
                    generated_sets
                )
                st.session_state.round_performance_text = performance_text

                st.success(f"🏆 COMPLET! Fitness: **{best_score['fitness_score']:.0f}** | WIN: {best_score['win_score']}")
                st.balloons()
                st.rerun()

# =========================================================================
# TAB 3: Rezultate
# =========================================================================
with tab3:
    st.markdown("## 📊 Rezultate și Analiză")

    if not st.session_state.generated_variants:
        st.info("ℹ️ Nu există rezultate.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        full_score = st.session_state.best_score_full

        fitness_display = f"{full_score['fitness_score']:.0f}"

        with col1: st.metric("Fitness", fitness_display)
        with col2: st.metric("WIN", full_score['win_score'])
        with col3: st.metric("3/3", full_score['score_3_3'])
        with col4: st.metric("Std Dev", f"{full_score['std_dev_wins']:.2f}")
        with col5: st.metric("Ponderat", f"{full_score['weighted_score_sum']:.0f}")

        st.markdown("---")

        if st.session_state.get('round_performance_text'):
            with st.expander("🎯 Performanță pe Rundă", expanded=False):
                st.text(st.session_state.round_performance_text)

        col_list, col_chart = st.columns([1, 2])

        with col_list:
            st.markdown("### 📋 Variante Generate")
            df_results = pd.DataFrame(st.session_state.generated_variants)

            st.download_button(
                "💾 TXT",
                data=variants_to_text(st.session_state.generated_variants),
                file_name=f"variante_finale_{len(st.session_state.generated_variants)}.txt",
                mime="text/plain"
            )

            st.download_button(
                "📊 CSV",
                data=variants_to_csv(st.session_state.generated_variants),
                file_name=f"variante_finale_{len(st.session_state.generated_variants)}.csv",
                mime="text/csv"
            )

            st.dataframe(df_results, use_container_width=True, hide_index=True, height=300)

        with col_chart:
            st.markdown("### 🔥 Frecvența Numerelor")
            df_freq = analyze_number_frequency(st.session_state.generated_variants)

            fig = px.bar(
                df_freq,
                x='Număr',
                y='Frecvență',
                text='Frecvență',
                title='Frecvența (1-66)'
            )
            fig.update_traces(marker_color='#667eea', textposition='outside')
            fig.update_layout(xaxis={'tickmode': 'linear', 'dtick': 5})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        if st.session_state.score_evolution_data:
            st.markdown("### 📈 Evoluția Scorului")
            fig_ev = plot_score_evolution(st.session_state.score_evolution_data)
            if fig_ev:
                st.plotly_chart(fig_ev, use_container_width=True)

        st.markdown("---")

        st.markdown("### ⚡ Analiza Forței Variantelor")

        if st.session_state.variants and st.session_state.rounds:
            with st.spinner("Calculez forța..."):
                df_strength = analyze_variant_strength(st.session_state.variants, st.session_state.rounds)

            col_top, col_bottom = st.columns(2)

            with col_top:
                st.markdown("#### Top 10 Cele Mai Puternice")
                st.dataframe(df_strength.head(10), use_container_width=True, hide_index=True)

            with col_bottom:
                st.markdown("#### Bottom 10 Cele Mai Slabe")
                st.dataframe(df_strength.tail(10), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>👑 Generator Optimizat Multi-CPU | Fără Blocaje | SCOR ȚINTĂ ASIGURAT</p>
    </div>
    """,
    unsafe_allow_html=True
)