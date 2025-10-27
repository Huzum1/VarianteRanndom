import streamlit as st
import pandas as pd
import random
import time
from copy import deepcopy
import plotly.express as px # Asigură-te că plotly este instalat (vezi requirements.txt)
import statistics
import numpy as np
from io import BytesIO
from multiprocessing import Pool, cpu_count
from functools import partial 
import sys 

# =========================================================================
# CONSTANTE ȘI CONFIGURARE PAGINĂ
# =========================================================================

# CONSTANTE DE VITEZĂ ȘI RESURSE
MAX_RANDOM_ATTEMPTS = 20000 
INTERMEDIATE_SAVE_INTERVAL = 5000 
NUM_PROCESSES = max(1, cpu_count() - 1) # Numărul de procese folosite în ambele faze
CHART_UPDATE_INTERVAL = 500 
LOCAL_SEARCH_DEFAULT_ITERATIONS = 20000 
# CONSTANTE DE SCOR
PENALTY_FACTOR_K = 0.5  
NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS = 20 

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
# FUNCȚII UTILITY
# =========================================================================

def clean_variant_combination(numbers_str):
    """Curăță șirul de numere, asigură unicitatea și le sortează."""
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
    """Parse variantele din text, curățând duplicatele."""
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
    """Procesează fișier TXT/CSV cu variante."""
    if file is None: return [], []
    try:
        content = file.getvalue().decode("utf-8")
        variants, errors, _, _ = parse_variants(content)
        return variants, errors
    except Exception as e:
        return [], [f"Eroare citire fișier: {str(e)}"]

def process_round_text(text):
    """Procesează textul rundelor (din fișier sau manual)."""
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
    """Procesează fișierul de runde (folosește cache)."""
    if rounds_file is None: return [], []
    try:
        content = rounds_file.getvalue().decode("utf-8")
        return process_round_text(content)
    except Exception as e:
        return [], []

def precompute_variant_sets(variants):
    """Precomputează set-urile pentru toate variantele (OPTIMIZARE MAJORĂ)."""
    variant_sets = []
    for v in variants:
        try:
            v_set = frozenset(int(n) for n in v['combination'].split() if n.isdigit())
            variant_sets.append(v_set)
        except:
            variant_sets.append(frozenset())
    return variant_sets

def get_round_weights(rounds):
    """Calculează ponderile exponențiale (Recency Weighting) pentru runde."""
    N = len(rounds)
    if N == 0: return {}
    BASE = 0.99 
    reversed_rounds = list(reversed(rounds))
    weights = [BASE**(N - i - 1) for i in range(N)]
    return {runda: weight for runda, weight in zip(reversed_rounds, weights)}


def calculate_wins_optimized(variant_indices, all_variant_sets, rounds_data, round_weights_data=None, use_deviation_penalty=False, penalty_factor_k=0.5):
    """Calculează scorul multi-obiectiv OPTIMIZAT. Acceptă liste de liste serializabile."""
    
    # Reconstruiește set-urile (CRITIC pentru siguranța pickling-ului)
    rounds = [frozenset(r) for r in rounds_data]
    if round_weights_data:
        # Reconstruiește cheile frozenset din cheile string
        round_weights = {frozenset(eval(k)): v for k, v in round_weights_data.items()}
    else:
        round_weights = {r: 1.0 for r in rounds}
        
    if not rounds or not variant_indices:
        return {
            'win_score': 0, 'score_3_3': 0, 'score_2_2': 0,
            'weighted_score_sum': 0, 'std_dev_wins': 0, 'fitness_score': -float('inf'),
            'score_per_round': {}
        }
    
    # Variantele sunt accesate direct din lista precomputată folosind indicii
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
        
        score_per_round[str(list(runda_set))] = {'wins': wins_in_round, '3_3': score_3_3_in_round, '2_2': score_2_2_in_round}
        
    wins_list = [
        (d['wins'] * 1000) + (d['3_3'] * 10) + (d['2_2'] * 1) 
        for d in score_per_round.values()
    ]
    
    std_dev_wins = statistics.stdev(wins_list) if len(wins_list) > 1 else 0

    base_score = weighted_score_sum if round_weights_data else total_wins
    base_score_multi = base_score * 100000 + total_3_3 * 100 + total_2_2 * 1
    
    fitness_score = base_score_multi
    
    if use_deviation_penalty:
         fitness_score = base_score_multi - (penalty_factor_k * std_dev_wins * 100000 * 10) 
    
    return {
        'win_score': total_wins,
        'score_3_3': total_3_3,
        'score_2_2': total_2_2,
        'weighted_score_sum': weighted_score_sum,
        'std_dev_wins': std_dev_wins,
        'fitness_score': fitness_score,
        'score_per_round': score_per_round
    }

def compare_scores(current_score, best_score, target_win_score):
    """Compară două scoruri folosind Fitness Score ca prioritate principală."""
    
    if current_score['fitness_score'] > best_score['fitness_score']: return True
    if current_score['fitness_score'] < best_score['fitness_score']: return False
        
    if current_score['win_score'] > best_score['win_score']: return True
    if current_score['win_score'] < best_score['win_score']: return False
        
    if current_score['score_3_3'] > best_score['score_3_3']: return True
    if current_score['score_3_3'] < best_score['score_3_3']: return False
        
    if current_score['score_2_2'] > best_score['score_2_2']: return True
        
    return False

def analyze_round_performance(generated_variants, rounds_set, variant_sets_precomputed=None):
    """Calculează performanța detaliată pe rundă."""
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
        
        runda_display = st.session_state.rounds_raw[i] if i < len(st.session_state.rounds_raw) else f"Runda {i+1} Set: {runda_set}"

        for v_set in variant_sets:
            matches = len(v_set.intersection(runda_set))
            if matches >= 4: wins_in_round += 1
            elif matches == 3: score_3_3_in_round += 1
            elif matches == 2: score_2_2_in_round += 1
            
        results_lines.append(f"{runda_display} - WINs: {wins_in_round}, 3/3: {score_3_3_in_round}, 2/2: {score_2_2_in_round}")
    return '\n'.join(results_lines)

def variants_to_text(variants):
    """Convertește lista de variante în format TXT (ID, numere), fără antet."""
    return '\n'.join([f"{v['id']},{v['combination']}" for v in variants])

def variants_to_csv(variants):
    """Convertește lista de variante în format CSV."""
    df = pd.DataFrame(variants)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue().decode('utf-8')

def analyze_number_frequency(variants):
    """Calculează frecvența fiecărui număr (1-66) în eșantion."""
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
    """Calculează scorul de "Forță" pentru fiecare variantă din pool-ul mare."""
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

# =========================================================================
# FUNCȚII PLOTLY
# =========================================================================

def plot_score_evolution(scores_data):
    """Generează graficul de evoluție a scorului Fitness."""
    if not scores_data:
        return None
    
    df = pd.DataFrame(scores_data)
    
    df['Fitness'] = df['Fitness'].astype(float)
    
    fig = px.line(
        df,
        x='Încercare',
        y='Fitness',
        title='Evoluția Scorului Fitness (Faza 1 Multi-CPU)',
        labels={'Încercare': 'Număr Încercări', 'Fitness': 'Scor Fitness (Ponderat)'}
    )
    fig.update_traces(line_color='#27ae60', line_width=2)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        font_color='white',
        xaxis_title='Număr Încercări',
        yaxis_title='Scor Fitness (Log)',
        yaxis_type='log'
    )
    return fig

# =========================================================================
# FUNCȚII MULTIPROCESSING (Cu Corecție PicklingError)
# =========================================================================

def evaluate_random_sample_worker(seed, count, num_variants, rounds_data, round_weights_data, use_deviation_penalty, penalty_factor_k, all_variant_sets):
    """
    Worker pentru evaluare paralelă Faza 1. 
    Toate datele mari sunt preluate prin functools.partial.
    
    Primește: doar Seed (pentru generarea aleatorie), și Indicii
    """
    
    random.seed(seed)
    indices = list(range(num_variants))
    random.shuffle(indices)
    sample_indices = indices[:count]
    
    # Apelăm funcția de scor cu set-urile mari care sunt deja "încărcate" în worker prin partial
    score = calculate_wins_optimized(
        sample_indices, 
        all_variant_sets, # Lista de frozenset-uri precomputate
        rounds_data, # Lista de liste simple
        round_weights_data, # Dict cu chei string
        use_deviation_penalty, 
        penalty_factor_k
    )
    
    return sample_indices, score

def evaluate_candidate_hole_worker(candidate_idx, all_variant_sets, weak_round_sets_data):
    """
    Worker pentru evaluarea candidaților pe rundele slabe (Faza 2).
    
    Argumentele complexe (all_variant_sets, weak_round_sets_data) sunt preluate 
    prin functools.partial.
    
    Returnează: (indice_candidat, hole_score)
    """
    
    # Recreează frozenset-urile (critice pentru siguranța Pickling)
    weak_round_sets = [frozenset(r) for r in weak_round_sets_data]

    # Folosim direct set-ul variantei din argumentul dat de `partial`
    variant_set = all_variant_sets[candidate_idx]
    hole_score = 0
    
    for runda_set in weak_round_sets:
        matches = len(variant_set.intersection(runda_set))
        if matches >= 4: hole_score += 1000
        elif matches == 3: hole_score += 100
        elif matches == 2: hole_score += 10
    
    return candidate_idx, hole_score


# =========================================================================
# STREAMLIT UI & LOGIC FLOW
# =========================================================================

st.markdown("# 👑 Generator Variante Loterie (Premium)")
st.markdown("### Optimizare Multi-CPU cu Uniformitate, Recență și Hole Coverage")

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
    
    st.metric(
        "Fitness (Ponderat)",
        fitness_display
    )
    st.caption(f"WIN: {full_score['win_score']} | 3/3: {full_score['score_3_3']} | Std Dev: {full_score['std_dev_wins']:.2f}")

    st.markdown("---")
    st.markdown("#### 💾 Salvari Intermediare")
    if st.session_state.intermediate_saves:
        st.success(f"🎉 {len(st.session_state.intermediate_saves)} eșantioane salvate")
        
        for i, save in enumerate(st.session_state.intermediate_saves):
            save_score_display = f"{save['score']['fitness_score']:.0f}" if save['score']['fitness_score'] != -float('inf') else "-inf"
            save_name = f"#{i+1} | WIN: {save['score']['win_score']} | Fitness: {save_score_display}"
            
            col_idx, col_dl = st.columns([2, 1])
            
            with col_idx:
                 st.caption(save_name)
                 
            with col_dl:
                st.download_button(
                    "💾 TXT",
                    data=variants_to_text(save['variants']