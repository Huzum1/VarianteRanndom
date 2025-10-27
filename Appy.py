import streamlit as st
import pandas as pd
import random
import time
from copy import deepcopy
import plotly.express as px
import statistics
import numpy as np
from io import BytesIO
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
import math

# =========================================================================
# CONSTANTE ȘI CONFIGURARE PAGINĂ (AJUSTATE PENTRU FLEXIBILITATE RUNDE)
# =========================================================================

# CONSTANTE DE VITEZĂ ȘI RESURSE
MAX_RANDOM_ATTEMPTS = 50000 
NUM_PROCESSES = max(1, cpu_count() - 1)
CHART_UPDATE_INTERVAL = 1000
EARLY_STOP_NO_IMPROVEMENT = 5000 

# CONSTANTE DE SCOR ȘI LOTERIE 
PENALTY_FACTOR_K = 0.5 
NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS = 10 
LOTTERY_MAX_NUMBER = 66
# NOU: Rundele sunt flexibile. Validăm MINIMUM 2 numere pentru a evita linii goale/inutile.
LOTTERY_ROUND_MIN_NUMBERS_FLEX = 2 
# VARIANTE: Minim 4 numere pentru o variantă de joc (6/66, 12/66, etc.)
LOTTERY_VARIANT_MIN_NUMBERS = 4 

st.set_page_config(
    page_title="Generator Variante Loterie (FLEXIBIL)",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZAT (Neschimbat) ---
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
    .perturbation-status { background-color: #9b59b6; color: white; }
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
if 'manual_rounds_input' not in st.session_state: st.session_state.manual_rounds_input = ""
if 'variants_input_text' not in st.session_state: st.session_state.variants_input_text = ""
if 'params' not in st.session_state:
    st.session_state.params = {
        'count': 1165,
        'target_wins_plus': 10,
        'local_search_iterations': 5000,
        'use_recency_weighting': True,
        'use_deviation_penalty': True,
        'penalty_factor_k': PENALTY_FACTOR_K
    }
if 'score_evolution_data' not in st.session_state:
    st.session_state.score_evolution_data = []
if 'round_performance_text' not in st.session_state:
    st.session_state.round_performance_text = ""
if 'rounds_parse_errors' not in st.session_state:
    st.session_state.rounds_parse_errors = []

# =========================================================================
# FUNCȚII UTILITY ȘI SCOR (LOGICA WIN PĂSTRATĂ)
# =========================================================================

def precompute_variant_sets(variants):
    """Calculează seturile de numere o singură dată."""
    sets = []
    for v in variants:
        try:
            numbers = [int(n) for n in v['combination'].split()]
            sets.append(frozenset(numbers))
        except:
            sets.append(frozenset()) 
    return sets

def calculate_match_score(variant_set, round_set):
    """Calculează scorul de potrivire (WIN/3/2)"""
    matches = len(variant_set.intersection(round_set))
    return matches

def get_round_weights(rounds):
    """Calculează o pondere inversă bazată pe recența rundelor."""
    weights = {}
    num_rounds = len(rounds)
    if num_rounds == 0:
        return {r: 1 for r in rounds}

    for i, r in enumerate(rounds):
        weight = 1.0 
        if num_rounds > 10:
             if i >= num_rounds - 10:
                 weight = 0.1 + 0.9 * (i - (num_rounds - 10)) / 9 
             else:
                 weight = 0.1
        else:
             weight = 0.1 + 0.9 * i / (num_rounds - 1) if num_rounds > 1 else 1.0
             
        weights[r] = weight
        
    return weights

def calculate_wins_optimized(indices, all_variant_sets, rounds_data_list, round_weights_data=None, 
                             use_deviation_penalty=True, penalty_factor_k=PENALTY_FACTOR_K):
    """Calculează scorul de Fitness complet pentru un set de indici.
    
    WIN logic: 4+ meciuri, 3/3, 2/2.
    """
    
    current_variant_sets = [all_variant_sets[i] for i in indices]
    
    total_win_score = 0
    total_score_3_3 = 0
    total_score_2_2 = 0
    weighted_score_sum = 0
    
    score_per_round = {}
    wins_per_round = []
    
    round_weights = {frozenset(r): w for r, w in round_weights_data} if round_weights_data else None

    for r_data in rounds_data_list:
        runda_set = frozenset(r_data)
        round_wins = 0
        round_3_3 = 0
        round_2_2 = 0
        
        weight = round_weights.get(runda_set, 1.0) if round_weights else 1.0
        
        for v_set in current_variant_sets:
            matches = calculate_match_score(v_set, runda_set)
            
            # 4+ matches are considered WINs for general coverage calculation
            if matches >= 4: round_wins += 1
            elif matches == 3: round_3_3 += 1
            elif matches == 2: round_2_2 += 1
        
        total_win_score += round_wins
        total_score_3_3 += round_3_3
        total_score_2_2 += round_2_2
        
        weighted_score_sum += round_wins * weight 
        
        wins_per_round.append(round_wins)
        
        score_per_round[str(r_data)] = {
            'wins': round_wins,
            '3_3': round_3_3,
            '2_2': round_2_2
        }
        
    std_dev_wins = statistics.stdev(wins_per_round) if len(wins_per_round) >= 2 else 0
    
    fitness_score = weighted_score_sum
    fitness_score += total_score_3_3 * 0.01  
    fitness_score += total_score_2_2 * 0.001 
    
    if use_deviation_penalty:
        penalty = penalty_factor_k * std_dev_wins 
        fitness_score -= penalty

    return {
        'win_score': total_win_score,
        'score_3_3': total_score_3_3,
        'score_2_2': total_score_2_2,
        'weighted_score_sum': weighted_score_sum,
        'std_dev_wins': std_dev_wins,
        'fitness_score': fitness_score,
        'score_per_round': score_per_round
    }

def compare_scores(score1, score2):
    """Compară două scoruri bazate pe Fitness-ul Ponderat."""
    if score1['fitness_score'] > score2['fitness_score']:
        return True
    return False

def plot_score_evolution(data):
    """Generează graficul de evoluție a scorului de Fitness."""
    if not data: return None
    df = pd.DataFrame(data)
    fig = px.line(df, x='Încercare', y='Fitness', title='Evoluția Scorului de Fitness')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)'), yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
    )
    return fig

def analyze_round_performance(generated_variants, rounds, generated_sets):
    """Analizează și formatează performanța finală pe rundă."""
    rounds_data_list = [sorted(list(r)) for r in rounds]
    performance_list = []
    for r_data in rounds_data_list:
        runda_set = frozenset(r_data)
        round_wins = 0
        round_3_3 = 0
        round_2_2 = 0
        for v_set in generated_sets:
            matches = calculate_match_score(v_set, runda_set)
            if matches >= 4: round_wins += 1
            elif matches == 3: round_3_3 += 1
            elif matches == 2: round_2_2 += 1
        
        performance_list.append({
            'Runda': ' '.join(map(str, r_data)),
            'WINs': round_wins,
            '3/3': round_3_3,
            '2/2': round_2_2
        })
    df = pd.DataFrame(performance_list)
    df_sorted = df.sort_values(by=['WINs', '3/3', '2/2'], ascending=[True, True, True])
    output = "Performanță Detaliată pe Rundă (Sortat de la cele mai slabe):\n\n"
    output += df_sorted.to_string(index=False)
    return output

# --- FUNCȚII DE PARSARE ȘI VALIDARE ---

def validate_variant_set(numbers, variant_id, min_numbers=LOTTERY_VARIANT_MIN_NUMBERS, max_value=LOTTERY_MAX_NUMBER):
    """Verifică dacă un set de numere (variante) este valid."""
    unique_numbers = sorted(list(set(numbers)))
    
    if len(unique_numbers) < min_numbers:
        return False, f"Varianta {variant_id} are sub {min_numbers} numere unice ({len(unique_numbers)})."
        
    for n in unique_numbers:
        if not (1 <= n <= max_value):
            return False, f"Varianta {variant_id} conține numărul {n} care nu este în intervalul 1-{max_value}."
            
    return True, unique_numbers

def clean_variant_combination(numbers_str):
    """Curăță șirul de numere, asigură unicitatea și le sortează."""
    try:
        parts = [p.strip() for p in numbers_str.split() if p.strip().isdigit()]
        valid_numbers = [int(p) for p in parts]
        unique_numbers = list(set(valid_numbers))
        unique_numbers.sort()
        duplicates_removed = len(valid_numbers) - len(unique_numbers)
        return unique_numbers, duplicates_removed
    except:
        return [], 0

def parse_variants(text):
    """Parse variantele din text."""
    variants = []
    errors = []
    total_internal_duplicates_removed = 0
    total_inter_duplicates_removed = 0
    lines = text.strip().split('\n')
    unique_combinations = set()
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue
        parts = line.split(',', 1)
        if len(parts) != 2:
            errors.append(f"Linia {i}: Format invalid (lipsește virgula)")
            continue
            
        variant_id = parts[0].strip()
        numbers_raw = parts[1].strip()
        if not variant_id or not numbers_raw: continue
        
        cleaned_numbers_list, duplicates_removed = clean_variant_combination(numbers_raw)
        total_internal_duplicates_removed += duplicates_removed
        
        is_valid, validation_result = validate_variant_set(cleaned_numbers_list, variant_id)
        
        if not is_valid:
            errors.append(f"Linia {i} (ID: {variant_id}): {validation_result}")
            continue
        
        cleaned_combination_str = ' '.join(map(str, validation_result))
        
        if cleaned_combination_str in unique_combinations:
            errors.append(f"Linia {i} (ID: {variant_id}): Combinație duplicată ({cleaned_combination_str}).")
            total_inter_duplicates_removed += 1
            continue
            
        unique_combinations.add(cleaned_combination_str)
        variants.append({'id': variant_id, 'combination': cleaned_combination_str})
    
    df = pd.DataFrame(variants)
    if not df.empty:
        df['id'] = (df.index + 1).astype(str)
        final_variants = df.to_dict('records')
    else:
        final_variants = []
    
    return final_variants, errors, total_internal_duplicates_removed, total_inter_duplicates_removed

def parse_variants_file(file):
    if file is None: return [], [], 0, 0
    try:
        content = file.getvalue().decode("utf-8")
        return parse_variants(content)
    except Exception as e:
        return [], [f"Eroare citire fișier: {str(e)}"], 0, 0

def process_round_text(text):
    """
    MODIFICAT: Procesează textul rundelor FĂRĂ restricție pe numărul de numere (doar 1-66).
    """
    rounds_set = set()
    rounds_raw_list = []
    parse_errors = []
    
    if not text:
        return [], [], []

    lines = text.strip().split('\n')
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue
        try:
            parts = line.split()
            numbers = [int(p.strip()) for p in parts if p.strip().isdigit()]
            
            unique_numbers = sorted(list(set(numbers)))

            # NOUA VALIDARE: DOAR MINIM 2 NUMERE ȘI ÎN INTERVAL (1-66)
            is_valid = (len(unique_numbers) >= LOTTERY_ROUND_MIN_NUMBERS_FLEX and 
                        all(1 <= n <= LOTTERY_MAX_NUMBER for n in unique_numbers))

            if is_valid:
                round_frozenset = frozenset(unique_numbers)
                
                if round_frozenset not in rounds_set:
                    rounds_set.add(round_frozenset)
                    rounds_raw_list.append(unique_numbers)
            else:
                parse_errors.append(f"Linia {i}: Invalidă. Min. {LOTTERY_ROUND_MIN_NUMBERS_FLEX} nr. în 1-{LOTTERY_MAX_NUMBER}. Găsit: {len(unique_numbers)} nr.")
        except Exception as e:
            parse_errors.append(f"Linia {i}: Eroare de parsare. Asigură-te că sunt numere. Detalii: {str(e)}")
            continue
            
    return list(rounds_set), rounds_raw_list, parse_errors

def parse_rounds_file(file):
    if file is None: return [], [], []
    try:
        content = file.getvalue().decode("utf-8")
        return process_round_text(content)
    except Exception as e:
        return [], [], [f"Eroare citire fișier: {str(e)}"]

# =========================================================================
# FUNCȚII MULTIPROCESSING (NESCHIMBATE)
# =========================================================================

def evaluate_random_sample_worker(args):
    # ... (unchanged) ...
    seed, count, num_variants, rounds_data, round_weights_data, \
        use_deviation_penalty, penalty_factor_k, all_variant_sets = args
    
    random.seed(seed)
    indices = list(range(num_variants))
    random.shuffle(indices)
    sample_indices = indices[:count]
    
    score = calculate_wins_optimized(
        sample_indices, all_variant_sets, rounds_data, round_weights_data,
        use_deviation_penalty, penalty_factor_k
    )
    return sample_indices, score

def evaluate_candidate_hole_worker(args):
    # ... (unchanged) ...
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
# STREAMLIT UI & LOGIC FLOW 
# =========================================================================

st.markdown("# 👑 Generator Variante Loterie (Runde FLEXIBILE)")
st.markdown("### Testează cu număr variabil de numere per rundă (1-66). Variante minime: 4 numere.")

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
    
    upload_method = st.radio("Metodă:", ["📄 Fișier", "⌨️ Text"], horizontal=True, key="upload_method_tab1")
    
    if upload_method == "📄 Fișier":
        variants_file = st.file_uploader(
            "Încarcă fișier (format: ID, numere separate prin virgulă)", 
            type=['txt', 'csv'],
            key="variants_file_uploader"
        )
        
        if variants_file:
            variants_list, errors, _, _ = parse_variants_file(variants_file) 
            
            if st.button("📥 Încarcă din Fișier", use_container_width=True, type="primary"):
                st.session_state.variants = variants_list
                
                if errors:
                    st.error(f"Erori de validare/duplicare găsite: {len(errors)} variante invalide/duplicate.")
                    for err in errors[:10]: st.write(f"- {err}")
                
                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 {len(st.session_state.variants)} variante **valide** încărcate! (Min. {LOTTERY_VARIANT_MIN_NUMBERS} numere)")
                    st.rerun()
                elif not errors: st.warning("Fișierul a fost citit, dar nu s-au găsit variante valide.")
    else:
        st.session_state.variants_input_text = st.text_area(
            "Variante (ID, numere)", 
            value=st.session_state.variants_input_text, 
            height=300,
            key="variants_input_text_area"
        )

        if st.button("📥 Încarcă din Text", use_container_width=True, type="primary"):
            if st.session_state.variants_input_text:
                variants_list, errors, _, _ = parse_variants(st.session_state.variants_input_text)
                st.session_state.variants = variants_list
                
                if errors:
                    st.error(f"Erori de validare/duplicare găsite: {len(errors)} variante invalide/duplicate.")
                    for err in errors[:10]: st.write(f"- {err}")
                
                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 {len(st.session_state.variants)} variante **valide** încărcate! (Min. {LOTTERY_VARIANT_MIN_NUMBERS} numere)")
                    st.rerun()
                elif not errors: st.warning("Textul a fost procesat, dar nu s-a găsit nicio variantă validă.")

    if st.session_state.variants:
        df_variants = pd.DataFrame(st.session_state.variants)
        st.markdown("### Previzualizare Variante Încărcate")
        st.dataframe(df_variants.head(10))

# =========================================================================
# TAB 2: Generare & Optimizare
# =========================================================================
with tab2:
    st.markdown("## 🎲 Pas 2: Configurare și Generare")
    
    if not st.session_state.variants:
        st.warning("⚠️ Încarcă variante mai întâi!")
    else:
        st.markdown("### 1. Încarcă Rundele (FLEXIBIL)")
        st.info(f"**Validare:** Fiecare rundă trebuie să conțină minim {LOTTERY_ROUND_MIN_NUMBERS_FLEX} numere unice în intervalul 1-{LOTTERY_MAX_NUMBER}.")
        
        col_file, col_manual = st.columns(2)
        rounds_file = col_file.file_uploader("Fișier Runde (Numere spațiate)", type=['txt', 'csv'], key="rounds_uploader")
        manual_rounds_input = col_manual.text_area("Sau adaugă manual (o rundă pe linie, numere spațiate)", value=st.session_state.manual_rounds_input, height=100, key="manual_rounds_input_area")
        st.session_state.manual_rounds_input = manual_rounds_input
        
        # LOGICA DE ÎNCĂRCARE ȘI COMBINARE 
        rounds_from_file_set, rounds_from_file_raw, errors_file = parse_rounds_file(rounds_file)
        rounds_from_manual_set, rounds_from_manual_raw, errors_manual = process_round_text(manual_rounds_input)

        st.session_state.rounds_parse_errors = errors_file + errors_manual
        
        all_rounds_set = set()
        rounds_raw_list = []
        
        for r_set, r_raw in zip(rounds_from_file_set, rounds_from_file_raw):
            if r_set not in all_rounds_set:
                all_rounds_set.add(r_set)
                rounds_raw_list.append(r_raw)
                
        for r_set, r_raw in zip(rounds_from_manual_set, rounds_from_manual_raw):
            if r_set not in all_rounds_set:
                all_rounds_set.add(r_set)
                rounds_raw_list.append(r_raw)
            
        st.session_state.rounds = list(all_rounds_set)
        st.session_state.rounds_raw = rounds_raw_list
        
        st.metric("Total Runde Unice", len(st.session_state.rounds))
        
        if st.session_state.rounds_parse_errors:
             st.error(f"Erori la parsarea rundelor: {len(st.session_state.rounds_parse_errors)} linii ignorate.")
             for err in st.session_state.rounds_parse_errors[:5]: st.caption(f"- {err}")
        
        if len(st.session_state.rounds) > 0 and st.checkbox("Afișează primele 10 runde încărcate"):
            st.code('\n'.join([' '.join(map(str, r)) for r in st.session_state.rounds_raw[:10]]))

        st.markdown("### 2. Configurare Algoritm")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.params['count'] = st.number_input(
                "Variante de generat", min_value=1, max_value=len(st.session_state.variants), 
                value=min(st.session_state.params['count'], len(st.session_state.variants)), key="variant_count_input"
            )
        
        with col2:
            st.session_state.params['target_wins_plus'] = st.number_input(
                "WINs Țintă +", min_value=0, value=st.session_state.params['target_wins_plus'], key="target_wins_input"
            )
            
        with col3:
            st.session_state.params['local_search_iterations'] = st.number_input(
                "Iterații Faza 2 (Vizual/Stop)", min_value=0, value=5000, key="local_iterations_input"
            )
            
        col4, col5 = st.columns(2)
        with col4:
            st.session_state.params['use_recency_weighting'] = st.checkbox(
                "Ponderare Recență", value=True, key="recency_weighting_checkbox"
            )
        with col5:
            st.session_state.params['use_deviation_penalty'] = st.checkbox(
                f"Penalizare Uniformitate (K={PENALTY_FACTOR_K})", value=True, key="deviation_penalty_checkbox"
            )
        
        st.session_state.params['penalty_factor_k'] = PENALTY_FACTOR_K 


        if st.button("🚀 GENEREAZĂ OPTIMIZAT", use_container_width=True, type="primary", key="generate_button"):
            
            if not st.session_state.rounds:
                st.error(f"Încarcă runde mai întâi! (Minim {LOTTERY_ROUND_MIN_NUMBERS_FLEX} numere, Max {LOTTERY_MAX_NUMBER})")
            else:
                # --- LOGICA DE OPTIMIZARE (PĂSTRATĂ) ---
                count = st.session_state.params['count']
                use_deviation_penalty = st.session_state.params['use_deviation_penalty']
                penalty_factor_k = st.session_state.params['penalty_factor_k']
                
                target_win_score = len(st.session_state.rounds) + st.session_state.params['target_wins_plus']
                rounds_data_list = [sorted(list(r)) for r in st.session_state.rounds]
                
                round_weights_data = None
                if st.session_state.params['use_recency_weighting']:
                    round_weights = get_round_weights(st.session_state.rounds)
                    round_weights_data = [(sorted(list(k)), v) for k, v in round_weights.items()]
                
                best_score = { 'win_score': 0, 'score_3_3': 0, 'score_2_2': 0,
                    'weighted_score_sum': 0, 'std_dev_wins': 0, 'fitness_score': -float('inf'), 'score_per_round': {}
                }
                best_variant_indices = []
                st.session_state.score_evolution_data = []
                
                if not st.session_state.variants_sets_precomputed:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                
                all_variant_sets = st.session_state.variants_sets_precomputed
                num_variants = len(st.session_state.variants)
                
                # =========================================================================
                # FAZA 1: Căutare Aleatorie Multi-CPU
                # =========================================================================
                st.info(f"🚀 FAZA 1: Căutare Aleatorie Multi-CPU ({MAX_RANDOM_ATTEMPTS:,} încercări)")
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                chart_placeholder = st.empty()
                score_evolution_data = []
                attempts, no_improvement_counter = 0, 0
                
                pool_faza1 = None
                try:
                    pool_faza1 = Pool(processes=NUM_PROCESSES)
                    seeds = [random.randint(0, 999999) + i for i in range(MAX_RANDOM_ATTEMPTS)]
                    args_list = [(seed, count, num_variants, rounds_data_list, round_weights_data, use_deviation_penalty, penalty_factor_k, all_variant_sets) for seed in seeds]
                    batch_size_faza1 = NUM_PROCESSES * 10
                    total_batches = (MAX_RANDOM_ATTEMPTS + batch_size_faza1 - 1) // batch_size_faza1
                    
                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size_faza1
                        end_idx = min(start_idx + batch_size_faza1, MAX_RANDOM_ATTEMPTS)
                        batch_args = args_list[start_idx:end_idx]
                        batch_results = pool_faza1.map(evaluate_random_sample_worker, batch_args)
                        
                        for sample_indices, current_score in batch_results:
                            attempts += 1
                            if compare_scores(current_score, best_score):
                                best_score = current_score.copy()
                                best_variant_indices = sample_indices
                                no_improvement_counter = 0
                            else:
                                no_improvement_counter += 1
                            
                            if attempts % 50 == 0 or attempts == 1 or no_improvement_counter == 0:
                                score_evolution_data.append({'Încercare': attempts, 'Fitness': best_score['fitness_score']})
                        
                        progress_bar.progress(min(1.0, attempts / MAX_RANDOM_ATTEMPTS))
                        score_detail = f"FIT: **{best_score['fitness_score']:.0f}** | WIN: {best_score['win_score']:,} | Țintă: {target_win_score}"
                        status_html = f"""<div class="status-box">FAZA 1: {attempts:,}/{MAX_RANDOM_ATTEMPTS:,} | No Improve: {no_improvement_counter}<div class="score-detail">{score_detail}</div></div>"""
                        status_placeholder.markdown(status_html, unsafe_allow_html=True)
                        
                        if attempts % CHART_UPDATE_INTERVAL == 0:
                            fig = plot_score_evolution(score_evolution_data)
                            if fig: chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        if no_improvement_counter >= EARLY_STOP_NO_IMPROVEMENT: break
                        if attempts >= MAX_RANDOM_ATTEMPTS: break
                
                except Exception as e:
                    st.error(f"🔴 EROARE CRITICĂ ÎN FAZA 1: {str(e)}")
                finally:
                    if pool_faza1: pool_faza1.close(); pool_faza1.join()
                        
                st.session_state.optimization_attempts = attempts
                st.session_state.score_evolution_data = score_evolution_data
                status_placeholder.empty()
                progress_bar.empty()
                st.success(f"✅ FAZA 1 Completă. Fitness inițial: {best_score['fitness_score']:.0f}")
                
                # =========================================================================
                # FAZA 2: Căutare Evolutivă Multi-CPU
                # =========================================================================

                if best_variant_indices:
                    st.info("🧬 FAZA 2: Evolutivă continuă (Robustness Focus).")
                    local_status_placeholder = st.empty()
                    local_progress = st.progress(0)
                    current_best_indices = best_variant_indices.copy()
                    current_best_score = best_score.copy()
                    no_improve_local, local_attempts = 0, 0
                    pool_faza2 = None
                    try:
                        while True:
                            local_attempts += 1
                            variant_scores = {}
                            for idx in current_best_indices:
                                score_single = calculate_wins_optimized([idx], all_variant_sets, rounds_data_list, round_weights_data, use_deviation_penalty=False, penalty_factor_k=0)
                                score_metric = (score_single['win_score'] * 10000 + score_single['score_3_3'] * 100 + score_single['score_2_2'])
                                variant_scores[idx] = score_metric
                            
                            if not variant_scores: break 
                            weakest_idx = min(variant_scores, key=variant_scores.get)
                            weakest_position = current_best_indices.index(weakest_idx)
                            
                            score_per_round_dict = {}
                            for k_str, v in current_best_score['score_per_round'].items():
                                try: k_list = eval(k_str); score_per_round_dict[frozenset(k_list)] = v
                                except: continue
                            
                            weakest_rounds = sorted(score_per_round_dict.items(), key=lambda item: (item[1]['wins'] * 1000 + item[1]['3_3'] * 10 + item[1]['2_2']))[:NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS]
                            weak_round_sets_data = [sorted(list(item[0])) for item in weakest_rounds]
                            available_indices = [i for i in range(num_variants) if i not in current_best_indices]
                            if not available_indices: break
                            
                            sample_size = min(200, len(available_indices))
                            sampled_indices = random.sample(available_indices, sample_size)
                            
                            if pool_faza2: pool_faza2.close(); pool_faza2.join()
                            pool_faza2 = Pool(processes=NUM_PROCESSES)
                            args_list_faza2 = [(idx, all_variant_sets, weak_round_sets_data) for idx in sampled_indices]
                            hole_scores = pool_faza2.map(evaluate_candidate_hole_worker, args_list_faza2)
                            
                            best_candidate_idx, max_hole_score = None, -1
                            for candidate_idx, hole_score in hole_scores:
                                if hole_score > max_hole_score: max_hole_score, best_candidate_idx = hole_score, candidate_idx
                                
                            if not best_candidate_idx: no_improve_local += 1
                            if no_improve_local > 1000: break
                            
                            test_indices = current_best_indices.copy()
                            test_indices[weakest_position] = best_candidate_idx
                            test_score = calculate_wins_optimized(test_indices, all_variant_sets, rounds_data_list, round_weights_data, use_deviation_penalty, penalty_factor_k)
                            
                            if compare_scores(test_score, current_best_score):
                                current_best_score, current_best_indices = test_score.copy(), test_indices.copy()
                                no_improve_local = 0
                            else:
                                no_improve_local += 1
                            
                            local_progress.progress(min(1.0, local_attempts / 100000)) 
                            score_detail = f"FIT: **{current_best_score['fitness_score']:.0f}** | WIN: {current_best_score['win_score']:,} / {target_win_score}"
                            status_html = f"""<div class="status-box local-search-status">FAZA 2: Încercare {local_attempts:,} | No Improve Streak: {no_improve_local}<div class="score-detail">{score_detail}</div></div>"""
                            local_status_placeholder.markdown(status_html, unsafe_allow_html=True)

                            if current_best_score['win_score'] >= target_win_score: break
                            if local_attempts >= 100000: break
                        
                    except Exception as e:
                        st.error(f"🔴 EROARE CRITICĂ ÎN FAZA 2: {str(e)}")
                    finally:
                        if pool_faza2: pool_faza2.close(); pool_faza2.join()
                            
                    best_score = current_best_score
                    best_variant_indices = current_best_indices.copy()
                    local_status_placeholder.empty()
                    local_progress.empty()
                    st.success(f"✅ FAZA 2 Finalizată. Fitness curent: {best_score['fitness_score']:.0f}")
                
                # =========================================================================
                # FAZA 3: Perturbare Globală
                # =========================================================================

                if best_variant_indices:
                    st.info("🌪️ FAZA 3: Perturbare Globală (Simulated Annealing Simplificat).")
                    force_status_placeholder = st.empty()
                    current_best_indices = best_variant_indices.copy()
                    current_best_score = best_score.copy()
                    
                    for perturb_attempts in range(1, MAX_RANDOM_ATTEMPTS + 1):
                        available_indices_force = [i for i in range(num_variants) if i not in current_best_indices]
                        if not available_indices_force: break
                        
                        variant_scores = {}
                        for idx in current_best_indices:
                            score_single = calculate_wins_optimized([idx], all_variant_sets, rounds_data_list, round_weights_data, use_deviation_penalty=False, penalty_factor_k=0)
                            score_metric = (score_single['win_score'] * 10000 + score_single['score_3_3'] * 100 + score_single['score_2_2'])
                            variant_scores[idx] = score_metric
                            
                        weakest_idx = min(variant_scores, key=variant_scores.get)
                        weakest_position = current_best_indices.index(weakest_idx)
                        candidate_idx = random.choice(available_indices_force)
                        
                        test_indices = current_best_indices.copy()
                        test_indices[weakest_position] = candidate_idx
                        test_score = calculate_wins_optimized(test_indices, all_variant_sets, rounds_data_list, round_weights_data, use_deviation_penalty, penalty_factor_k)
                        
                        is_better = compare_scores(test_score, current_best_score)
                        accept_worse = False
                        if not is_better and current_best_score['fitness_score'] != -float('inf'):
                            current_temperature = 0.01 + 0.09 * (1 - perturb_attempts / MAX_RANDOM_ATTEMPTS)
                            score_diff = current_best_score['fitness_score'] - test_score['fitness_score']
                            if score_diff > 0:
                                if (random.random() < current_temperature): accept_worse = True
                                
                        if is_better or accept_worse:
                            current_best_score, current_best_indices = test_score.copy(), test_indices.copy()
                            perturb_type = "Îmbunătățire" if is_better else f"Acceptare Inferioară (Temp: {current_temperature:.2f})"
                        else:
                            perturb_type = "Respinsă"
                            
                        score_detail = f"FIT: **{current_best_score['fitness_score']:.0f}** | WIN: {current_best_score['win_score']:,} / {target_win_score}"
                        status_html = f"""<div class="status-box perturbation-status">FAZA 3: Încercare {perturb_attempts:,}/{MAX_RANDOM_ATTEMPTS} | Schimbare: {perturb_type}<div class="score-detail">{score_detail}</div></div>"""
                        force_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                        
                        if current_best_score['win_score'] >= target_win_score: break
                        if perturb_attempts >= MAX_RANDOM_ATTEMPTS: break
                        
                    best_variant_indices = current_best_indices.copy()
                    force_status_placeholder.empty()
                    best_score = current_best_score
                    st.success(f"✅ FAZA 3 Finalizată. Fitness final: {best_score['fitness_score']:.0f}")
                    
                st.session_state.local_search_attempts = local_attempts + perturb_attempts
                st.session_state.generated_variants = [st.session_state.variants[i] for i in best_variant_indices]
                st.session_state.best_score_full = best_score
                
                generated_sets = [all_variant_sets[i] for i in best_variant_indices]
                performance_text = analyze_round_performance(st.session_state.generated_variants, st.session_state.rounds, generated_sets)
                st.session_state.round_performance_text = performance_text
                
                st.success(f"🏆 COMPLET! Fitness: **{best_score['fitness_score']:.0f}** | WIN: {best_score['win_score']}")
                st.balloons()
                st.rerun()

# =========================================================================
# TAB 3: Rezultate
# =========================================================================

with tab3:
    st.markdown("## 📊 Rezultate & Analiză Finală")

    if not st.session_state.generated_variants:
        st.info("Rulați optimizarea mai întâi în Tab-ul 2.")
    else:
        st.markdown("### 1. Setul de Variante Optimizat")
        
        df_results = pd.DataFrame(st.session_state.generated_variants)
        st.dataframe(df_results)

        csv_export = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Descărcă Variantele (CSV)", csv_export, "variante_optimizate_flexi.csv", "text/csv", key='download-csv')
        
        st.markdown("---")
        st.markdown("### 2. Performanța Finală")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        score = st.session_state.best_score_full
        
        with col_res1: st.metric("Fitness Final", f"{score['fitness_score']:.0f}")
        with col_res2: st.metric("Total WINs (4+ meciuri)", f"{score['win_score']:,}")
        with col_res3: st.metric("Total 3/3", f"{score['score_3_3']:,}")
        with col_res4: st.metric("Deviație Std", f"{score['std_dev_wins']:.2f}")

        st.markdown("---")
        st.markdown("### 3. Detaliu Performanță pe Rundă (Robustness)")

        st.text_area("Analiză Detaliată a Performanței", st.session_state.round_performance_text, height=400, key="performance_text_area")
        
        st.markdown("---")
        st.markdown("### 4. Evoluția Scorului")
        
        if st.session_state.score_evolution_data:
            fig = plot_score_evolution(st.session_state.score_evolution_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nicio evoluție disponibilă.")
