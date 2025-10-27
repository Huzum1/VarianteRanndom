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

# =========================================================================
# CONSTANTE ȘI CONFIGURARE PAGINĂ
# =========================================================================

# MODIFICARE CHEIE 1: Setăm limita de căutare aleatorie la 25.000
MAX_RANDOM_ATTEMPTS = 25000 
INTERMEDIATE_SAVE_INTERVAL = 5000 
PENALTY_FACTOR_K = 0.5  
NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS = 20 # Mărim pentru o acoperire mai bună
NUM_PROCESSES = max(1, cpu_count() - 1)  
CHART_UPDATE_INTERVAL = 500 # Actualizăm mai des graficul
LOCAL_SEARCH_DEFAULT_ITERATIONS = 20000 # MODIFICARE CHEIE 2: Mărim iteratiile Fazei 2

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


def calculate_wins_optimized(variant_indices, all_variant_sets, rounds, round_weights=None, use_deviation_penalty=False, penalty_factor_k=0.5):
    """Calculează scorul multi-obiectiv OPTIMIZAT."""
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
    
    if round_weights is None: 
        round_weights = {r: 1.0 for r in rounds}

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
        
        score_per_round[runda_set] = {'wins': wins_in_round, '3_3': score_3_3_in_round, '2_2': score_2_2_in_round}
        
    wins_list = [
        (d['wins'] * 1000) + (d['3_3'] * 10) + (d['2_2'] * 1) 
        for d in score_per_round.values()
    ]
    
    std_dev_wins = statistics.stdev(wins_list) if len(wins_list) > 1 else 0

    base_score = weighted_score_sum if round_weights else total_wins
    base_score_multi = base_score * 100000 + total_3_3 * 100 + total_2_2 * 1
    
    fitness_score = base_score_multi
    
    if use_deviation_penalty:
         # Penalizarea este aplicată pe deviația scorului compozit.
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
    # Eliminam comparatia cu target_win_score in Faza 1 pentru a trece rapid in Faza 2
    # if current_score['win_score'] >= target_win_score and best_score['win_score'] < target_win_score: 
    #     return True 
    
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
# FUNCȚII PLOTLY (PENTRU GRAFICUL ÎN TIMP REAL)
# =========================================================================

def plot_score_evolution(scores_data):
    """Generează graficul de evoluție a scorului Fitness."""
    if not scores_data:
        return None
    
    df = pd.DataFrame(scores_data)
    
    # Asigură-te că coloana 'Fitness' este numerică și formatată
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
# FUNCȚII MULTIPROCESSING (OPTIMIZARE PERFORMANȚĂ)
# =========================================================================

def evaluate_random_sample_worker(args):
    """Worker pentru evaluare paralelă (rulează într-un proces separat)."""
    seed, count, num_variants, rounds_data, round_weights_data, use_deviation_penalty, penalty_factor_k, all_variant_sets = args
    
    # Recreează frozenset-urile (care nu sunt picklable ușor) în interiorul workerului
    rounds = [frozenset(r) for r in rounds_data] 
    
    if round_weights_data:
        # Converteste cheile string înapoi la frozenset
        # Folosim eval() pentru a transforma stringul '[x, y, z]' în listă, apoi în frozenset
        round_weights = {frozenset(eval(k)): v for k, v in round_weights_data.items()}
    else:
        round_weights = None

    random.seed(seed)
    indices = list(range(num_variants))
    random.shuffle(indices)
    sample_indices = indices[:count]
    
    score = calculate_wins_optimized(
        sample_indices, all_variant_sets, rounds, 
        round_weights, use_deviation_penalty, penalty_factor_k
    )
    
    return sample_indices, score


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
    st.metric("CPU-uri Disponibile", NUM_PROCESSES)
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
                    data=variants_to_text(save['variants']),
                    file_name=f"salvare_{i+1}_A{save['attempt']}.txt",
                    mime="text/plain",
                    key=f"dl_save_{i}",
                    use_container_width=True
                )
    else:
        st.info("Nicio salvare intermediară.")

    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante & Curăță", "🎲 Generează & Optimizează", "📊 Rezultate & Analiză"])

# =========================================================================
# TAB 1: ÎNCARCĂ VARIANTE 
# =========================================================================
with tab1:
    st.markdown("## 📝 Pas 1: Încarcă Variantele Tale & Curăță Duplicatele")
    
    upload_method = st.radio("Metodă de încărcare:", ["📄 Fișier TXT/CSV", "⌨️ Text Manual"], horizontal=True)
    
    if upload_method == "📄 Fișier TXT/CSV":
        variants_file = st.file_uploader(
            "Încarcă fișier cu variante (format: ID, numere separate prin spațiu)", 
            type=['txt', 'csv'],
            key="variants_file_uploader"
        )
        
        if variants_file:
            variants_from_file, errors_from_file = parse_variants_file(variants_file)
            
            if st.button("📥 Încarcă & Curăță din Fișier", use_container_width=True, type="primary"):
                st.session_state.variants = variants_from_file
                
                if errors_from_file:
                    st.error("Au fost găsite erori:")
                    for err in errors_from_file[:10]:
                        st.write(f"- {err}")
                
                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 **{len(st.session_state.variants)}** variante unice încărcate din fișier!")
                    st.rerun()
                else:
                    st.warning("Nicio variantă validă în fișier.")
    else:
        st.session_state.variants_input_text = st.text_area(
            "Variante (ID, număr1 număr2 număr3...)", 
            value=st.session_state.variants_input_text, 
            height=300, 
            placeholder="Exemplu:\n1, 5 7 44 32 18\n2, 10 12 14 30 40"
        )

        if st.button("📥 Încarcă & Curăță din Text", use_container_width=True, type="primary"):
            if st.session_state.variants_input_text:
                variants_list, errors, total_internal_dup, total_inter_dup = parse_variants(st.session_state.variants_input_text)
                
                st.session_state.variants = variants_list
                
                if errors:
                    st.error("Au fost găsite erori de formatare:")
                    for err in errors[:10]:
                        st.write(f"- {err}")
                
                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 **{len(st.session_state.variants)}** variante unice încărcate!")
                    st.caption(f"Eliminate: {total_inter_dup} duplicate combinații, {total_internal_dup} duplicate numere interne.")
                    st.rerun()
                else:
                    st.warning("Nicio variantă validă după curățare.")
            else:
                st.warning("Introduceți variante înainte de a apăsa 'Încarcă'.")
    
    st.markdown("---")
    if st.session_state.variants:
        col_txt, col_csv = st.columns(2)
        with col_txt:
            st.download_button(
                "💾 Descarcă Variante Curățate (TXT)",
                data=variants_to_text(st.session_state.variants),
                file_name=f"variante_curatate_{len(st.session_state.variants)}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_csv:
            st.download_button(
                "📊 Descarcă Variante Curățate (CSV)",
                data=variants_to_csv(st.session_state.variants),
                file_name=f"variante_curatate_{len(st.session_state.variants)}.csv",
                mime="text/csv",
                use_container_width=True
            )

# =========================================================================
# TAB 2: Generare & Optimizare
# =========================================================================
with tab2:
    st.markdown("## 🎲 Pas 2: Configurare și Optimizare Evolutivă Multi-CPU")
    
    if not st.session_state.variants:
        st.warning("⚠️ Nu există variante curățate încă! Mergi la tab-ul 'Încarcă Variante & Curăță'.")
    else:
        st.markdown("### 1. Încarcă Rundele (Extragerile) de Bază")
        
        col_file, col_manual = st.columns(2)
        rounds_file = col_file.file_uploader("Încărcați fișierul cu Rundele", type=['txt', 'csv'], key="rounds_uploader")
        manual_rounds_input = col_manual.text_area("Sau adaugă runde manual", value=st.session_state.manual_rounds_input, height=100, placeholder="Exemplu:\n1 5 7 12 44 49")
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
        
        st.metric("Total Runde Unice Încărcate", len(st.session_state.rounds))
        
        if st.session_state.rounds_raw and st.session_state.get('round_performance_text'):
            st.markdown("#### 🎯 Performanța Eșantionului pe Rundă")
            performance_html = '<br>'.join([f"<p>{line}</p>" for line in st.session_state.round_performance_text.split('\n')])
            st.markdown(f'<div class="results-box">{performance_html}</div>', unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("### 2. Configurare Algoritm Multi-CPU")

        col_count, col_targets, col_iterations = st.columns(3)
        
        with col_count:
            st.markdown(f"Ai **{len(st.session_state.variants)}** variante disponibile.")
            st.session_state.params['count'] = st.number_input(
                "Câte variante să generez?", 
                min_value=1, 
                max_value=len(st.session_state.variants), 
                value=min(st.session_state.params['count'], len(st.session_state.variants)), 
                step=1, 
                key='param_count'
            )
        
        with col_targets:
            st.session_state.params['target_wins_plus'] = st.number_input(
                "WINs Țintă Suplimentare (+X)", 
                min_value=0, 
                value=st.session_state.params['target_wins_plus'], 
                step=1, 
                key='param_target'
            )
            st.caption(f"Target WIN: {len(st.session_state.rounds) + st.session_state.params['target_wins_plus']}")
            
        with col_iterations:
            st.session_state.params['local_search_iterations'] = st.number_input(
                "Iterații Căutare Evolutivă (Faza 2)", 
                min_value=0, 
                # MODIFICARE CHEIE 2: Folosim valoarea mărită
                value=LOCAL_SEARCH_DEFAULT_ITERATIONS, 
                step=100, 
                key='param_local_iter'
            )
            
        st.markdown("#### ⚙️ Mecanisme de Fitness Avansate")
        col_weight, col_dev, col_k = st.columns(3)
        
        with col_weight:
            st.session_state.params['use_recency_weighting'] = st.checkbox(
                "Ponderare după Recență", 
                value=st.session_state.params['use_recency_weighting'], 
                help="Prioritate rundelor recente", 
                key='param_weight'
            )
            
        with col_dev:
            st.session_state.params['use_deviation_penalty'] = st.checkbox(
                "Penalizare Uniformitate", 
                value=st.session_state.params['use_deviation_penalty'], 
                help="Uniformizează distribuția câștigurilor", 
                key='param_dev'
            )
        
        with col_k:
            st.session_state.params['penalty_factor_k'] = st.number_input(
                "Factor Penalizare K", 
                min_value=0.0, 
                value=st.session_state.params['penalty_factor_k'], 
                step=0.1, 
                format="%.2f", 
                disabled=not st.session_state.params['use_deviation_penalty'], 
                key='param_k'
            )

        if st.button("🚀 Generează & Optimizează (Multi-CPU)", use_container_width=True, type="primary"):
            
            if not st.session_state.rounds:
                st.error("Vă rugăm să încărcați runde mai întâi.")
            else:
                count = st.session_state.params['count']
                target_win_score = len(st.session_state.rounds) + st.session_state.params['target_wins_plus']
                local_search_iterations = st.session_state.params['local_search_iterations']
                use_recency_weighting = st.session_state.params['use_recency_weighting']
                use_deviation_penalty = st.session_state.params['use_deviation_penalty']
                penalty_factor_k = st.session_state.params['penalty_factor_k']
                
                rounds_data_list = [list(r) for r in st.session_state.rounds]
                
                if use_recency_weighting:
                    round_weights = get_round_weights(st.session_state.rounds)
                    round_weights_data = {str(list(k)): v for k, v in round_weights.items()}
                else:
                    round_weights_data = None
                
                attempts, local_attempts = 0, 0
                best_score = deepcopy(st.session_state.best_score_full)
                best_variant_indices = []
                st.session_state.intermediate_saves = [] 
                st.session_state.score_evolution_data = []
                
                if not st.session_state.variants_sets_precomputed:
                    with st.spinner("Precomputare set-uri variante..."):
                        st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                
                all_variant_sets = st.session_state.variants_sets_precomputed
                num_variants = len(st.session_state.variants)
                
                # =========================================================================
                # FAZA 1: Căutare Aleatorie PARALELIZATĂ (cu actualizare în TIMP REAL)
                # =========================================================================
                st.info(f"🚀 FAZA 1 (Multi-CPU cu {NUM_PROCESSES} procese): Max: {MAX_RANDOM_ATTEMPTS:,} încercări forțate.")

                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                chart_placeholder = st.empty()

                score_evolution_data = []
                
                with st.spinner(f"Optimizare paralelă pe {NUM_PROCESSES} CPU-uri..."):
                    
                    worker_args = [
                        (
                            random.randint(0, 1000000 + i),
                            count,
                            num_variants,
                            rounds_data_list, 
                            round_weights_data, 
                            use_deviation_penalty,
                            penalty_factor_k,
                            all_variant_sets
                        )
                        for i in range(MAX_RANDOM_ATTEMPTS)
                    ]
                    
                    pool = None
                    try:
                        pool = Pool(processes=NUM_PROCESSES)
                        results_iterator = pool.imap_unordered(evaluate_random_sample_worker, worker_args)
                        
                        for sample_indices, current_score in results_iterator:
                            attempts += 1
                            
                            is_better = compare_scores(current_score, best_score, target_win_score)

                            if is_better:
                                best_score = current_score.copy()
                                best_variant_indices = sample_indices
                            
                            # 1. ACTUALIZARE GRAFIC
                            if attempts % 100 == 0 or attempts == 1 or is_better:
                                score_evolution_data.append({
                                    'Încercare': attempts,
                                    'Fitness': best_score['fitness_score']
                                })
                            
                            if attempts % CHART_UPDATE_INTERVAL == 0 or attempts == MAX_RANDOM_ATTEMPTS:
                                fig = plot_score_evolution(score_evolution_data)
                                if fig:
                                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                            
                            # 2. ACTUALIZARE UI STATUS & PROGRES
                            
                            if attempts % 500 == 0 or attempts == MAX_RANDOM_ATTEMPTS:
                                progress_percent = min(1.0, attempts / MAX_RANDOM_ATTEMPTS)
                                progress_bar.progress(progress_percent)
                                
                                score_detail = f"FITNESS: **{best_score['fitness_score']:.0f}** | WIN: {best_score['win_score']:,} | 3/3: {best_score['score_3_3']:,} | StdDev: {best_score['std_dev_wins']:.2f}"
                                status_html = f"""
                                <div class="status-box">
                                    FAZA 1 (MULTI-CPU): {attempts:,}/{MAX_RANDOM_ATTEMPTS:,} încercări
                                    <div class="score-detail">{score_detail}</div>
                                </div>
                                """
                                status_placeholder.markdown(status_html, unsafe_allow_html=True)
                                
                            # 3. LOGICĂ DE SALVARE INTERMEDIARĂ
                            if attempts % INTERMEDIATE_SAVE_INTERVAL == 0:
                                save_variants = [st.session_state.variants[i] for i in best_variant_indices]
                                st.session_state.intermediate_saves.append({
                                    'attempt': attempts,
                                    'score': best_score.copy(),
                                    'variants': deepcopy(save_variants)
                                })
                                st.sidebar.caption(f"Salvare la {attempts:,} încercări.")
                                
                            # MODIFICARE CHEIE 3: Forțăm oprirea după MAX_RANDOM_ATTEMPTS.
                            if attempts >= MAX_RANDOM_ATTEMPTS:
                                break
                    
                    finally:
                        if pool:
                            pool.terminate()
                            pool.join()
                        
                st.session_state.optimization_attempts = attempts
                st.session_state.score_evolution_data = score_evolution_data
                status_placeholder.empty()
                progress_bar.empty()
                
                # =========================================================================
                # FAZA 2: Căutare Evolutivă (Hole Coverage)
                # =========================================================================

                if best_variant_indices and local_search_iterations > 0:
                    st.info(f"FAZA 2 (Evolutivă): Hole Coverage cu **{local_search_iterations:,}** iterații.")
                    
                    local_status_placeholder = st.empty()
                    current_best_indices = best_variant_indices.copy()
                    current_best_score = best_score.copy()
                    
                    round_weights_local = get_round_weights(st.session_state.rounds) if use_recency_weighting else None
                    
                    for local_attempts in range(1, local_search_iterations + 1):
                        
                        # 1. Identificare varianta cea mai slabă din eșantion
                        variant_scores = {}
                        for idx in current_best_indices:
                            score_single = calculate_wins_optimized(
                                [idx], all_variant_sets, st.session_state.rounds, round_weights=round_weights_local
                            )
                            score_metric = score_single['win_score'] * 10000 + score_single['score_3_3'] * 100 + score_single['score_2_2']
                            variant_scores[idx] = score_metric
                        
                        weakest_idx = min(variant_scores, key=variant_scores.get)
                        weakest_position = current_best_indices.index(weakest_idx)
                        
                        # 2. Identificare rundele cele mai slab acoperite (găurile)
                        weakest_rounds = sorted(
                            current_best_score['score_per_round'].items(),
                            key=lambda item: item[1]['wins'] * 1000 + item[1]['3_3'] * 10 + item[1]['2_2']
                        )[:NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS]
                        weak_round_sets = [item[0] for item in weakest_rounds]
                        
                        # 3. Caută cel mai bun candidat din pool-ul mare (eșantionare 1000)
                        best_hole_score = -1
                        best_candidate_idx = None
                        
                        available_indices = [i for i in range(num_variants) if i not in current_best_indices]
                        sample_size = min(1000, len(available_indices))
                        sampled_indices = random.sample(available_indices, sample_size)
                        
                        for candidate_idx in sampled_indices:
                            variant_set = all_variant_sets[candidate_idx]
                            hole_score = 0
                            
                            for runda_set in weak_round_sets:
                                matches = len(variant_set.intersection(runda_set))
                                if matches >= 4: hole_score += 1000
                                elif matches == 3: hole_score += 100
                                elif matches == 2: hole_score += 10
                            
                            if hole_score > best_hole_score:
                                best_hole_score = hole_score
                                best_candidate_idx = candidate_idx
                        
                        if best_candidate_idx is None:
                            best_candidate_idx = random.choice(available_indices)
                        
                        # 4. Testare și acceptare
                        test_indices = current_best_indices.copy()
                        test_indices[weakest_position] = best_candidate_idx
                        
                        test_score = calculate_wins_optimized(
                            test_indices, all_variant_sets, st.session_state.rounds,
                            round_weights_local, use_deviation_penalty, penalty_factor_k
                        )
                        
                        if compare_scores(test_score, current_best_score, target_win_score):
                            current_best_score = test_score.copy()
                            current_best_indices = test_indices.copy()
                        
                        # Actualizare status UI în timp real
                        if local_attempts % 500 == 0 or local_attempts == local_search_iterations:
                            score_detail = f"FITNESS: **{current_best_score['fitness_score']:.0f}** | WIN: {current_best_score['win_score']:,} | 3/3: {current_best_score['score_3_3']:,} | StdDev: {current_best_score['std_dev_wins']:.2f}"
                            local_status_html = f"""
                            <div class="status-box local-search-status">
                                FAZA 2 (EVOLUTIVĂ): {local_attempts:,}/{local_search_iterations:,}
                                <div class="score-detail">{score_detail}</div>
                            </div>
                            """
                            local_status_placeholder.markdown(local_status_html, unsafe_allow_html=True)
                    
                    best_score = current_best_score
                    best_variant_indices = current_best_indices.copy()
                    local_status_placeholder.empty()
                    st.success(f"FAZA 2 Finalizată. Fitness: **{best_score['fitness_score']:.0f}** după {local_attempts:,} iterații.")
                else:
                    local_attempts = 0

                st.session_state.local_search_attempts = local_attempts
                st.session_state.generated_variants = [st.session_state.variants[i] for i in best_variant_indices]
                st.session_state.best_score_full = best_score
                
                generated_sets = [all_variant_sets[i] for i in best_variant_indices]
                performance_text = analyze_round_performance(
                    st.session_state.generated_variants, 
                    st.session_state.rounds,
                    generated_sets
                )
                st.session_state.round_performance_text = performance_text
                
                st.success(f"🏆 Optimizare finalizată! Fitness: **{best_score['fitness_score']:.0f}**")
                st.balloons()
                st.rerun()
        
        if st.session_state.optimization_attempts > 0 or st.session_state.local_search_attempts > 0:
            st.info(f"Ultima rulare: **{st.session_state.optimization_attempts:,}** încercări aleatorii (multi-CPU) și **{st.session_state.local_search_attempts:,}** evolutive.")
            
            if st.session_state.score_evolution_data:
                st.markdown("---")
                st.markdown("### Evoluția Scorului (Ultima Rulare)")
                fig = plot_score_evolution(st.session_state.score_evolution_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Rezultate & Analiză
with tab3:
    st.markdown("## 📊 Rezultate și Analiză Ultra Premium")
    
    if not st.session_state.generated_variants:
        st.info("ℹ️ Nu există rezultate generate încă.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        full_score = st.session_state.best_score_full
        
        with col1: 
            fitness_display = f"{full_score['fitness_score']:.0f}" if full_score['fitness_score'] != -float('inf') else "-inf"
            st.metric("Fitness Final", fitness_display)
        with col2: st.metric("Scor WIN", full_score['win_score'])
        with col3: st.metric("Scor 3/3", f"{full_score['score_3_3']:,}")
        with col4: st.metric("Ecart Tip", f"{full_score['std_dev_wins']:.2f}")
        with col5: st.metric("Scor Ponderat", f"{full_score['weighted_score_sum']:.0f}")
        
        st.markdown("---")
        
        col_list, col_chart = st.columns([1, 2])
        
        with col_list:
            st.markdown("### 📋 Lista de Variante")
            df_results = pd.DataFrame(st.session_state.generated_variants)
            
            st.download_button(
                "💾 Descarcă TXT", 
                data=variants_to_text(st.session_state.generated_variants), 
                file_name=f"variante_optim_{len(st.session_state.generated_variants)}.txt", 
                mime="text/plain", 
                use_container_width=True
            )
            
            st.download_button(
                "📊 Descarcă CSV", 
                data=variants_to_csv(st.session_state.generated_variants), 
                file_name=f"variante_optim_{len(st.session_state.generated_variants)}.csv", 
                mime="text/csv", 
                use_container_width=True
            )
            
            st.dataframe(df_results, use_container_width=True, hide_index=True, height=250)
        
        with col_chart:
            st.markdown("### 🔥 Frecvența Numerelor (1-66)")
            df_freq = analyze_number_frequency(st.session_state.generated_variants)
            
            try:
                fig = px.bar(
                    df_freq,
                    x='Număr',
                    y='Frecvență',
                    text='Frecvență',
                    title='Frecvența de Apariție (1-66)',
                    labels={'Număr': 'Număr', 'Frecvență': 'Apariții'}
                )
                fig.update_traces(marker_color='#667eea', textposition='outside')
                fig.update_layout(xaxis={'tickmode': 'linear', 'dtick': 5}, yaxis={'tickformat': ','})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Frecvența medie așteptată: {(len(df_results) * 6) / 66:.2f} apariții per număr.")
            except Exception as e:
                st.warning(f"Eroare la generarea graficului Plotly: {e}. Asigurați-vă că 'plotly' este instalat.")

        st.markdown("---")
        
        st.markdown("### ⚡ Analiza Forței Variantelor în Pool")
        
        if st.session_state.variants and st.session_state.rounds:
            with st.spinner("Calculez forța variantelor..."):
                df_strength = analyze_variant_strength(st.session_state.variants, st.session_state.rounds)
            
            col_top, col_bottom = st.columns(2)
            
            with col_top:
                st.markdown("#### Top 10 Variante (Cele mai Puternice)")
                st.dataframe(df_strength.head(10), use_container_width=True, hide_index=True)
            
            with col_bottom:
                st.markdown("#### Bottom 10 Variante (Cele mai Slabe)")
                st.dataframe(df_strength.tail(10), use_container_width=True, hide_index=True)
            
            st.caption("Clasament bazat pe performanța istorică a fiecărei variante din pool.")
        else:
            st.info("Încărcați variante și runde pentru Analiza de Forță.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>👑 Generator Variante Loterie Multi-CPU | Optimizare Evolutivă Avansată (1-66)</p>
    </div>
    """,
    unsafe_allow_html=True
)
