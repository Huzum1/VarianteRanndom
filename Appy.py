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

# CONSTANTE DE VITEZĂ ȘI RESURSE - OPTIMIZATE
MAX_RANDOM_ATTEMPTS = 50000 
INTERMEDIATE_SAVE_INTERVAL = 10000
NUM_PROCESSES = max(1, cpu_count() - 1)
CHART_UPDATE_INTERVAL = 1000
LOCAL_SEARCH_DEFAULT_ITERATIONS = 5000 # Rămâne ca referință inițială/vizuală
BATCH_SIZE = 200 
EARLY_STOP_NO_IMPROVEMENT = 5000 
MAX_PERTURB_ITERATIONS = 5000 # Limita de siguranță Faza 3

# CONSTANTE DE SCOR ȘI OPTIMIZARE (Ajustat pentru Robustness)
PENALTY_FACTOR_K = 0.5  # Mărit de la 0.3 la 0.5 pentru a forța uniformitatea (robustness)
NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS = 10 
LOTTERY_MAX_NUMBER = 66
LOTTERY_MIN_NUMBERS = 4 # Număr minim de numere per variantă

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
    .perturbation-status { background-color: #9b59b6; color: white; } /* Nou pentru Faza 3 */
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# INITIALIZARE SESIUNE (fără modificări)
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
# FUNCȚII UTILITY ȘI SCOR (Modificat pentru Validare Robustă)
# =========================================================================

def validate_variant_set(numbers, variant_id, min_numbers=LOTTERY_MIN_NUMBERS, max_value=LOTTERY_MAX_NUMBER):
    """Verifică dacă un set de numere este valid."""
    unique_numbers = sorted(list(set(numbers)))
    
    # 1. Verifică numărul minim de numere
    if len(unique_numbers) < min_numbers:
        return False, f"Varianta {variant_id} are sub {min_numbers} numere unice ({len(unique_numbers)})."
        
    # 2. Verifică intervalul de numere
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
    """Parse variantele din text, curățând duplicatele și validând."""
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

# ... (Restul funcțiilor de utilitate și scor rămân neschimbate) ...

# =========================================================================
# FUNCȚII MULTIPROCESSING (fără modificări structurale)
# =========================================================================

def evaluate_random_sample_worker(args):
    seed, count, num_variants, rounds_data, round_weights_data, \
        use_deviation_penalty, penalty_factor_k, all_variant_sets = args
    
    random.seed(seed)
    indices = list(range(num_variants))
    random.shuffle(indices)
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
        # ... (Logica de afișare salvări) ...
    else:
        st.info("Nicio salvare.")

    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante", "🎲 Generează", "📊 Rezultate"])

# =========================================================================
# TAB 1: ÎNCARCĂ VARIANTE (Modificat pentru Validare)
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
            variants_list, errors, _, _ = parse_variants_file(variants_file) 
            
            if st.button("📥 Încarcă din Fișier", use_container_width=True, type="primary"):
                st.session_state.variants = variants_list
                
                if errors:
                    st.error(f"Erori de validare găsite: {len(errors)} variante invalide/duplicate.")
                    for err in errors[:10]:
                        st.write(f"- {err}")
                    if len(errors) > 10:
                        st.write(f"... și alte {len(errors) - 10} erori.")
                
                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 {len(st.session_state.variants)} variante **valide** încărcate!")
                    st.rerun()
                elif not errors:
                     st.warning("Fișierul a fost citit, dar nu s-au găsit variante valide.")
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
                
                if errors:
                    st.error(f"Erori de validare găsite: {len(errors)} variante invalide/duplicate.")
                    for err in errors[:10]:
                        st.write(f"- {err}")
                    if len(errors) > 10:
                        st.write(f"... și alte {len(errors) - 10} erori.")
                
                if st.session_state.variants:
                    st.session_state.variants_sets_precomputed = precompute_variant_sets(st.session_state.variants)
                    st.success(f"🎉 {len(st.session_state.variants)} variante **valide** încărcate!")
                    st.rerun()
                elif not errors:
                     st.warning("Textul a fost procesat, dar nu s-a găsit nicio variantă validă.")

    # ... (Restul Tab 1 - Download) ...

# =========================================================================
# TAB 2: Generare & Optimizare (Modificări pentru Faza 2 și Faza 3)
# =========================================================================
with tab2:
    st.markdown("## 🎲 Pas 2: Configurare și Generare")
    
    if not st.session_state.variants:
        st.warning("⚠️ Încarcă variante mai întâi!")
    else:
        st.markdown("### 1. Încarcă Rundele")
        
        # ... (Logica de încărcare a rundelor - neschimbată) ...
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
                help="Această limită este doar vizuală și pentru progres. Căutarea continuă până la scorul țintă sau blocaj de Fitness."
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
        
        # Setează K-factorul final
        st.session_state.params['penalty_factor_k'] = PENALTY_FACTOR_K 


        if st.button("🚀 GENEREAZĂ OPTIMIZAT", use_container_width=True, type="primary"):
            
            if not st.session_state.rounds:
                st.error("Încarcă runde mai întâi!")
            else:
                count = st.session_state.params['count']
                local_search_iterations = st.session_state.params['local_search_iterations']
                use_recency_weighting = st.session_state.params['use_recency_weighting']
                use_deviation_penalty = st.session_state.params['use_deviation_penalty']
                penalty_factor_k = st.session_state.params['penalty_factor_k']
                
                target_win_score = len(st.session_state.rounds) + st.session_state.params['target_wins_plus']
                
                rounds_data_list = [sorted(list(r)) for r in st.session_state.rounds]
                
                round_weights_data = None
                if use_recency_weighting:
                    round_weights = get_round_weights(st.session_state.rounds)
                    round_weights_data = [(sorted(list(k)), v) for k, v in round_weights.items()]
                
                attempts, local_attempts, perturb_attempts = 0, 0, 0
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
                
                # FAZA 1: Căutare Aleatorie Multi-CPU
                st.info(f"🚀 FAZA 1: {MAX_RANDOM_ATTEMPTS:,} încercări (Țintă WIN: {target_win_score})")

                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                chart_placeholder = st.empty()

                score_evolution_data = []
                no_improvement_counter = 0
                
                pool_faza1 = None
                try:
                    pool_faza1 = Pool(processes=NUM_PROCESSES)
                    # ... (Logica Fazei 1 - Neschimbată) ...
                    seeds = [random.randint(0, 999999) + i for i in range(MAX_RANDOM_ATTEMPTS)]
                    args_list = [
                        (seed, count, num_variants, rounds_data_list, round_weights_data,
                         use_deviation_penalty, penalty_factor_k, all_variant_sets)
                        for seed in seeds
                    ]
                    
                    batch_size = NUM_PROCESSES * 10
                    total_batches = (MAX_RANDOM_ATTEMPTS + batch_size - 1) // batch_size
                    
                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, MAX_RANDOM_ATTEMPTS)
                        batch_args = args_list[start_idx:end_idx]
                        
                        batch_results = pool_faza1.map(evaluate_random_sample_worker, batch_args)
                        
                        for sample_indices, current_score in batch_results:
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
                             # ... (Logica de salvare intermediară) ...
                             save_variants = [st.session_state.variants[i] for i in best_variant_indices]
                             st.session_state.intermediate_saves.append({
                                 'attempt': attempts,
                                 'score': best_score.copy(),
                                 'variants': deepcopy(save_variants)
                             })
                        
                        if no_improvement_counter >= EARLY_STOP_NO_IMPROVEMENT:
                            st.info(f"⏹️ Early stop Faza 1 la {attempts:,} (no improvement: {no_improvement_counter})")
                            break
                        
                        if attempts >= MAX_RANDOM_ATTEMPTS:
                             break
                
                except Exception as e:
                    st.error(f"Eroare Faza 1: {str(e)}")
                finally:
                    if pool_faza1:
                        pool_faza1.close()
                        pool_faza1.join()
                        
                st.session_state.optimization_attempts = attempts
                st.session_state.score_evolution_data = score_evolution_data
                status_placeholder.empty()
                progress_bar.empty()
                
                st.success(f"✅ FAZA 1 Completă: {attempts:,} încercări | Fitness: {best_score['fitness_score']:.0f}")
                
                # =========================================================================
                # FAZA 2: Căutare Evolutivă Multi-CPU (Robustness & Uniformity Focus)
                # =========================================================================

                if best_variant_indices:
                    st.info(f"🧬 FAZA 2: Evolutivă continuă. **Obiectiv: Maximizare Fitness (Uniformitate & Acoperire)**. Țintă WIN: {target_win_score}.")
                    
                    local_status_placeholder = st.empty()
                    local_progress = st.progress(0)
                    
                    current_best_indices = best_variant_indices.copy()
                    current_best_score = best_score.copy()
                    
                    no_improve_local = 0
                    local_attempts = 0
                    
                    pool_faza2 = None
                    try:
                        
                        # BUCĂ ÎNIFINITĂ: Se oprește la scorul țintă sau la Limita de Siguranță sau la Blocaj
                        while True:
                            local_attempts += 1
                            
                            # 1. Găsește varianta cea mai slabă (pe baza scorului de Fitness)
                            variant_scores = {}
                            for idx in current_best_indices:
                                score_single = calculate_wins_optimized(
                                    [idx], all_variant_sets, rounds_data_list, round_weights_data, 
                                    use_deviation_penalty=False, penalty_factor_k=0
                                )
                                score_metric = (score_single['win_score'] * 10000 + 
                                              score_single['score_3_3'] * 100 + 
                                              score_single['score_2_2'])
                                variant_scores[idx] = score_metric
                            
                            if not variant_scores: break 
                            
                            weakest_idx = min(variant_scores, key=variant_scores.get)
                            weakest_position = current_best_indices.index(weakest_idx)
                            
                            # 2. Identifică rundele slabe și candidații (Hole Coverage)
                            score_per_round_dict = {}
                            for k_str, v in current_best_score['score_per_round'].items():
                                try:
                                    k_list = eval(k_str)
                                    score_per_round_dict[frozenset(k_list)] = v
                                except:
                                    continue
                            
                            weakest_rounds = sorted(
                                score_per_round_dict.items(),
                                key=lambda item: (item[1]['wins'] * 1000 + 
                                                item[1]['3_3'] * 10 + 
                                                item[1]['2_2'])
                            )[:NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS]
                            
                            weak_round_sets = [item[0] for item in weakest_rounds]
                            weak_round_sets_data = [sorted(list(r)) for r in weak_round_sets]
                            
                            # Găsește candidați (Sample)
                            available_indices = [i for i in range(num_variants) 
                                               if i not in current_best_indices]
                            if not available_indices: break
                            
                            sample_size = min(BATCH_SIZE, len(available_indices))
                            sampled_indices = random.sample(available_indices, sample_size)
                            
                            # EVALUARE PARALELĂ (Hole Coverage)
                            args_list_faza2 = [
                                (idx, all_variant_sets, weak_round_sets_data)
                                for idx in sampled_indices
                            ]
                            
                            # Reinițializarea Pool-ului pentru stabilitate
                            if pool_faza2:
                                pool_faza2.close()
                                pool_faza2.join()
                            pool_faza2 = Pool(processes=NUM_PROCESSES)
                            
                            hole_scores = pool_faza2.map(evaluate_candidate_hole_worker, args_list_faza2)
                            
                            best_candidate_idx = None
                            max_hole_score = -1
                            for candidate_idx, hole_score in hole_scores:
                                if hole_score > max_hole_score:
                                    max_hole_score = hole_score
                                    best_candidate_idx = candidate_idx
                                
                            # 3. Testare și acceptare (Pe scorul de Fitness complet)
                            if not best_candidate_idx: 
                                no_improve_local += 1
                                if no_improve_local > 1000:
                                    st.info(f"⏹️ Faza 2 s-a blocat la {local_attempts:,} (1000 iterații fără îmbunătățire Fitness). Trecere la Faza 3 (Perturbare).")
                                    break
                                continue 
                                
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
                            
                            # 4. VERIFICARE SCOR ȚINTĂ ȘI UI UPDATE
                            local_progress.progress(min(1.0, local_attempts / local_search_iterations))
                            score_detail = f"FIT: **{current_best_score['fitness_score']:.0f}** | WIN: {current_best_score['win_score']:,} / {target_win_score}"
                            status_html = f"""
                            <div class="status-box local-search-status">
                                FAZA 2: Încercare {local_attempts:,} | No Improve Streak: {no_improve_local}
                                <div class="score-detail">{score_detail}</div>
                            </div>
                            """
                            local_status_placeholder.markdown(status_html, unsafe_allow_html=True)

                            if current_best_score['win_score'] >= target_win_score:
                                st.success(f"🎯 **SCOR ȚINTĂ ATINS ÎN FAZA 2!** WIN: {current_best_score['win_score']:,} la iterația {local_attempts:,}.")
                                break
                            
                            if local_attempts >= 100000: 
                                st.warning(f"⚠️ Faza 2 a atins limita de siguranță de 100k iterații. Trecere la Faza 3 (Perturbare).")
                                break
                        
                    except Exception as e:
                        st.error(f"🔴 EROARE CRITICĂ ÎN FAZA 2 (Multiprocessing Pool): {str(e)}")
                        st.warning("Se folosește cel mai bun rezultat obținut până la eroare.")
                    finally:
                        if pool_faza2:
                            pool_faza2.close()
                            pool_faza2.join()
                            
                    best_score = current_best_score
                    best_variant_indices = current_best_indices.copy()
                    local_status_placeholder.empty()
                    local_progress.empty()
                    
                    st.success(f"✅ FAZA 2 Finalizată. Fitness curent: {best_score['fitness_score']:.0f}")
                
                # =========================================================================
                # FAZA 3: Perturbare Globală (Simulated Annealing Simplificat)
                # =========================================================================

                if best_variant_indices:
                    
                    st.info(f"🌪️ FAZA 3: Perturbare Globală. **Obiectiv: Maximizare Fitness & Robustness**. Ieșirea din optimul local.")
                    
                    force_status_placeholder = st.empty()
                    perturb_attempts = 0
                    current_best_indices = best_variant_indices.copy()
                    current_best_score = best_score.copy()
                    
                    for perturb_attempts in range(1, MAX_PERTURB_ITERATIONS + 1):
                        
                        # 1. Găsește o pereche de indici de schimb
                        available_indices_force = [i for i in range(num_variants) 
                                                   if i not in current_best_indices]
                        
                        if not available_indices_force: break
                        
                        # Alege o varianta slabă de scos (cea mai slabă)
                        variant_scores = {}
                        for idx in current_best_indices:
                            score_single = calculate_wins_optimized(
                                [idx], all_variant_sets, rounds_data_list, round_weights_data, 
                                use_deviation_penalty=False, penalty_factor_k=0
                            )
                            score_metric = (score_single['win_score'] * 10000 + 
                                          score_single['score_3_3'] * 100 + 
                                          score_single['score_2_2'])
                            variant_scores[idx] = score_metric
                            
                        weakest_idx = min(variant_scores, key=variant_scores.get)
                        weakest_position = current_best_indices.index(weakest_idx)
                        
                        # Alege o varianta de introdus (aleatoriu, pentru perturbare)
                        candidate_idx = random.choice(available_indices_force)
                        
                        # 2. Testare Perturbare
                        test_indices = current_best_indices.copy()
                        test_indices[weakest_position] = candidate_idx
                        
                        test_score = calculate_wins_optimized(
                            test_indices, all_variant_sets, rounds_data_list,
                            round_weights_data, use_deviation_penalty, penalty_factor_k
                        )
                        
                        # 3. Acceptare Schimbare (Simulated Annealing Simplificat)
                        is_better = compare_scores(test_score, current_best_score)
                        
                        accept_worse = False
                        if not is_better and current_best_score['fitness_score'] != -float('inf'):
                            # Temperatura scade de la 10% la 1%
                            current_temperature = 0.01 + 0.09 * (1 - perturb_attempts / MAX_PERTURB_ITERATIONS)
                            score_diff = current_best_score['fitness_score'] - test_score['fitness_score']
                            
                            # Probabilitate invers proportionala cu diferenta negativa
                            if score_diff > 0:
                                # Acceptă o soluție mai slabă dacă e mică și probabilitatea e mare
                                # Folosim o probabilitate simplificată bazată pe temperatură.
                                if (random.random() < current_temperature):
                                    accept_worse = True
                                
                        if is_better or accept_worse:
                            current_best_score = test_score.copy()
                            current_best_indices = test_indices.copy()
                            
                            if is_better:
                                perturb_type = "Îmbunătățire"
                            elif accept_worse:
                                perturb_type = f"Acceptare Inferioară (Temp: {current_temperature:.2f})"
                            
                        else:
                            perturb_type = "Respinsă"
                            
                        # 4. UI Update
                        score_detail = f"FIT: **{current_best_score['fitness_score']:.0f}** | WIN: {current_best_score['win_score']:,} / {target_win_score}"
                        status_html = f"""
                        <div class="status-box perturbation-status">
                            FAZA 3: Încercare {perturb_attempts:,}/{MAX_PERTURB_ITERATIONS} | Schimbare: {perturb_type}
                            <div class="score-detail">{score_detail}</div>
                        </div>
                        """
                        force_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                        
                        if current_best_score['win_score'] >= target_win_score:
                            st.success(f"🎯 **SCOR ȚINTĂ ATINS ÎN FAZA 3!** WIN: {current_best_score['win_score']:,} la iterația {perturb_attempts:,}.")
                            break
                        
                        if perturb_attempts >= MAX_PERTURB_ITERATIONS:
                            st.info(f"⏹️ Faza 3 a atins limita maximă de {MAX_PERTURB_ITERATIONS} iterații.")
                            break
                        
                    best_variant_indices = current_best_indices.copy()
                    force_status_placeholder.empty()
                    
                    # Re-evaluarea finală
                    best_score = current_best_score
                    st.success(f"✅ FAZA 3 Finalizată. Fitness final: {best_score['fitness_score']:.0f}")
                    
                st.session_state.local_search_attempts = local_attempts + perturb_attempts
                st.session_state.generated_variants = [st.session_state.variants[i] for i in best_variant_indices]
                st.session_state.best_score_full = best_score
                
                # Analiză performanță
                generated_sets = [all_variant_sets[i] for i in best_variant_indices]
                performance_text = analyze_round_performance(
                    st.session_state.generated_variants, 
                    st.session_state.rounds,
                    generated_sets
                )
                st.session_state.round_performance_text = performance_text
                
                st.success(f"🏆 COMPLET! Fitness: **{best_score['fitness_score']:.0f}** | WIN: {best_score['win_score']}")
                st.balloons()
                st.rerun()

# ... (Restul Tab 3: Rezultate - Neschimbat) ...
