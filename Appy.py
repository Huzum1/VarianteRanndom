import streamlit as st
import pandas as pd
import random
import time
from copy import deepcopy
import plotly.express as px
import statistics
import numpy as np
from io import BytesIO

# =========================================================================
# CONSTANTE ȘI CONFIGURARE PAGINĂ
# =========================================================================

MAX_RANDOM_ATTEMPTS = 100000
INTERMEDIATE_SAVE_INTERVAL = 5000 
PENALTY_FACTOR_K = 0.5  
NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS = 10 

st.set_page_config(
    page_title="Generator Variante Loterie (Premium)",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZAT (Păstrat) ---
st.markdown("""
    <style>
    /* Stiluri generale păstrate */
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #667eea; }
    h1 { color: white !important; text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem; }
    h2, h3 { color: #667eea !important; }
    /* Stil pentru chenarul cu rezultate (Performanță Rundă) */
    .results-box { border: 1px solid #764ba2; padding: 15px; border-radius: 8px; background-color: #333333; color: white; height: 400px; overflow-y: scroll; font-family: monospace; }
    /* Stil pentru chenarul de status compact în timpul optimizării (Păstrat) */
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
# INITIALIZARE SESIUNE ȘI FUNCȚII UTILITY (Păstrate)
# =========================================================================

# Inițializare Session State (Păstrată)
if 'variants' not in st.session_state: st.session_state.variants = []
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
if 'variants_input_text' not in st.session_state: st.session_state.variants_input_text = "" # NOU: Păstrează input-ul variantelor
if 'params' not in st.session_state:
    st.session_state.params = {
        'count': 1165,
        'target_wins_plus': 10,
        'local_search_iterations': 5000,
        'use_recency_weighting': True,
        'use_deviation_penalty': True,
        'penalty_factor_k': PENALTY_FACTOR_K
    }

# --- Funcțiile Utility (Păstrate) ---

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
        # Păstrează doar combinațiile unice
        df_unique = df.drop_duplicates(subset=['combination']).reset_index(drop=True)
        # Reatribuie ID-uri secvențiale
        df_unique['id'] = (df_unique.index + 1).astype(str)
        final_variants = df_unique.to_dict('records')
        total_inter_duplicates_removed = len(variants) - len(final_variants)
    else:
        final_variants = []
        total_inter_duplicates_removed = 0
    
    return final_variants, errors, total_internal_duplicates_removed, total_inter_duplicates_removed

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
            rounds_set_list.append(round_numbers)
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
        # st.error(f"Eroare la procesarea fișierului de runde: {e}") # Comentat pt stabilitate
        return [], []

def variants_to_text(variants):
    return '\n'.join([f"{v['id']},{v['combination']}" for v in variants])

def variants_to_csv(variants):
    # ATENȚIE: Metodă robustă de export CSV (fără argumente problematice ca line_terminator)
    df = pd.DataFrame(variants)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue().decode('utf-8')

# --- Alte funcții de calcul (calculate_wins, compare_scores, etc.) rămân neschimbate ---

# ... [funcțiile rămase rămân neschimbate: get_round_weights, calculate_wins, compare_scores, analyze_round_performance, analyze_number_frequency, analyze_variant_strength]


# =========================================================================
# STREAMLIT UI & LOGIC FLOW (Modificată Tab-ul 1)
# =========================================================================

# Header
st.markdown("# 👑 Generator Variante Loterie (Premium)")
st.markdown("### Optimizare pe Uniformitate, Recență și Hole Coverage")

# Sidebar
# ... (Logică Sidebar - Păstrată) ...
with st.sidebar:
    st.markdown("## 📊 Statistici Curente")
    st.metric("Variante Curățate", len(st.session_state.variants))
    st.metric("Runde Încărcate", len(st.session_state.rounds_raw))
    st.markdown("---")
    st.markdown("#### Scor Final Obținut")
    
    full_score = st.session_state.best_score_full
    st.metric(
        "Fitness (Ponderat)",
        f"{full_score['fitness_score']:.0f}"
    )
    st.caption(f"WIN: {full_score['win_score']:,} | 3/3: {full_score['score_3_3']:,} | Std Dev: {full_score['std_dev_wins']:.2f}")

    st.markdown("---")
    st.markdown("#### Salvari Intermediare")
    if st.session_state.intermediate_saves:
        st.success(f"💾 {len(st.session_state.intermediate_saves)} eșantioane salvate")
        
        all_saves_text = ""
        for i, save in enumerate(st.session_state.intermediate_saves):
            all_saves_text += f"=== Salvare Intermediară #{i+1} | WIN: {save['score']['win_score']} | Fitness: {save['score']['fitness_score']:.0f} | Încercare: {save['attempt']} ===\n"
            all_saves_text += variants_to_text(save['variants']) + "\n\n"
            
        st.download_button(
            "💾 Descarcă Toate Salvările (TXT)",
            data=all_saves_text,
            file_name="salvari_intermediare_optimizare.txt",
            mime="text/plain",
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
# TAB 1: ÎNCARCĂ VARIANTE (LOGICĂ RESTABILITĂ)
# =========================================================================
with tab1:
    st.markdown("## 📝 Pas 1: Încarcă Variantele Tale & Curăță Duplicatele")
    st.session_state.variants_input_text = st.text_area(
        "Variante (ID, număr1 număr2 număr3...)", 
        value=st.session_state.variants_input_text, 
        height=300, 
        placeholder="Exemplu:\n1, 5 7 44 32 18\n2, 10 12 14 30 40"
    )

    col_load, col_download_curatate = st.columns([1, 1])
    
    if col_load.button("📥 Încarcă & Curăță Variante", use_container_width=True, type="primary"):
        if st.session_state.variants_input_text:
            
            # --- LOGICA DE ÎNCĂRCARE AICI ---
            variants_list, errors, total_internal_duplicates_removed, total_inter_duplicates_removed = parse_variants(st.session_state.variants_input_text)
            
            st.session_state.variants = variants_list
            
            if errors:
                st.error("Au fost găsite erori de formatare sau variante invalide:")
                for err in errors:
                    st.write(f"- {err}")
            
            if st.session_state.variants:
                st.success(f"🎉 **{len(st.session_state.variants)}** variante unice au fost încărcate și curățate.")
                st.caption(f"S-au eliminat {total_inter_duplicates_removed} duplicate la nivel de combinație și {total_internal_duplicates_removed} duplicate la nivel de numere interne.")
            else:
                 st.warning("Nicio variantă validă nu a fost încărcată după curățare.")
            
            st.rerun() # Reîmprospătează Sidebar
        else:
            st.warning("Vă rugăm să introduceți variante înainte de a apăsa 'Încarcă & Curăță'.")
    
    with col_download_curatate:
        if st.session_state.variants:
            st.download_button(
                "💾 Descarcă Variante Curățate (TXT)",
                data=variants_to_text(st.session_state.variants),
                file_name=f"variante_curatate_{len(st.session_state.variants)}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Descarcă Variante Curățate (Nu există date)")


# =========================================================================
# TAB 2: Generare & Optimizare (Păstrată)
# =========================================================================
with tab2:
    st.markdown("## 🎲 Pas 2: Configurare și Optimizare Evolutivă")
    
    if not st.session_state.variants:
        st.warning("⚠️ Nu există variante curățate încă! Mergi la tab-ul 'Încarcă Variante & Curăță'.")
    else:
        # --- ÎNCĂRCARE RUNDE (Păstrată) ---
        st.markdown("### 1. Încarcă Rundele (Extragerile) de Bază")
        
        col_file, col_manual = st.columns(2)
        rounds_file = col_file.file_uploader("Încărcați fișierul cu Rundele (extragerile)", type=['txt', 'csv'], key="rounds_uploader")
        manual_rounds_input = col_manual.text_area("Sau adaugă runde manual", value=st.session_state.manual_rounds_input, height=100, placeholder="Exemplu:\n1 5 7 12 44 49")
        st.session_state.manual_rounds_input = manual_rounds_input
        
        rounds_from_file_set, rounds_from_file_raw = parse_rounds_file(rounds_file)
        rounds_from_manual_set, rounds_from_manual_raw = process_round_text(manual_rounds_input)

        all_rounds_set_dict = {}
        for r_set, r_raw in zip(rounds_from_file_set, rounds_from_file_raw): all_rounds_set_dict[frozenset(r_set)] = r_raw
        for r_set, r_raw in zip(rounds_from_manual_set, rounds_from_manual_raw): all_rounds_set_dict[frozenset(r_set)] = r_raw
            
        st.session_state.rounds = list(all_rounds_set_dict.keys())
        st.session_state.rounds_raw = list(all_rounds_set_dict.values())
        
        st.metric("Total Runde Unice Încărcate", len(st.session_state.rounds))
        
        if st.session_state.rounds_raw and st.session_state.get('round_performance_text'):
            st.markdown("#### 🎯 Performanța Eșantionului pe Rundă")
            performance_html = '<br>'.join([f"<p>{line}</p>" for line in st.session_state.round_performance_text.split('\n')])
            st.markdown(f'<div class="results-box">{performance_html}</div>', unsafe_allow_html=True)
            st.markdown("---")

        
        # --- CONFIGURARE OPTIMIZARE (Păstrată) ---
        st.markdown("### 2. Configurare Algoritm (Setări Ultra Premium)")
        # ... (Restul configurației din TAB 2 și logica de optimizare - Păstrată, inclusiv revenirea la statusul simplificat) ...

        col_count, col_targets, col_iterations = st.columns(3)
        
        with col_count:
            st.markdown(f"Ai **{len(st.session_state.variants)}** variante unice disponibile.")
            st.session_state.params['count'] = st.number_input("Câte variante să generez pe eșantion?", min_value=1, max_value=len(st.session_state.variants), value=st.session_state.params['count'], step=1, key='param_count')
        
        with col_targets:
            st.session_state.params['target_wins_plus'] = st.number_input("WINs (>=4/4) Țintă Suplimentare (+X)", min_value=0, value=st.session_state.params['target_wins_plus'], step=1, key='param_target')
            st.caption(f"Targetul WIN (brut) este: {len(st.session_state.rounds) + st.session_state.params['target_wins_plus']}")
            
        with col_iterations:
            st.session_state.params['local_search_iterations'] = st.number_input("Iterații Căutare Evolutivă (Locală)", min_value=0, value=st.session_state.params['local_search_iterations'], step=100, key='param_local_iter')
            
        st.markdown("#### ⚙️ Mecanisme de Fitness Avansate")
        col_weight, col_dev, col_k = st.columns(3)
        
        with col_weight:
            st.session_state.params['use_recency_weighting'] = st.checkbox("Ponderare după Recență (Exponențială)", value=st.session_state.params['use_recency_weighting'], help="Dă prioritate potrivirilor obținute în rundele cele mai recente.", key='param_weight')
            
        with col_dev:
            st.session_state.params['use_deviation_penalty'] = st.checkbox("Penalizare Ecart-Tip (Uniformitate)", value=st.session_state.params['use_deviation_penalty'], help="Penalizează eșantioanele cu câștiguri concentrate în puține runde (favorizează uniformitatea).", key='param_dev')
        
        with col_k:
            st.session_state.params['penalty_factor_k'] = st.number_input("Factor Penalizare K", min_value=0.0, value=st.session_state.params['penalty_factor_k'], step=0.1, format="%.2f", disabled=not st.session_state.params['use_deviation_penalty'], key='param_k')


        if st.button("🚀 Generează & Optimizează (Evolutiv)", use_container_width=True, type="primary"):
            
            if not st.session_state.rounds:
                st.error("Vă rugăm să încărcați sau să introduceți runde mai întâi.")
            else:
                
                # Colectare Parametri și inițializare
                count = st.session_state.params['count']
                target_win_score = len(st.session_state.rounds) + st.session_state.params['target_wins_plus']
                local_search_iterations = st.session_state.params['local_search_iterations']
                use_recency_weighting = st.session_state.params['use_recency_weighting']
                use_deviation_penalty = st.session_state.params['use_deviation_penalty']
                penalty_factor_k = st.session_state.params['penalty_factor_k']
                round_weights = get_round_weights(st.session_state.rounds) if use_recency_weighting else None
                
                attempts, local_attempts = 0, 0
                best_score = deepcopy(st.session_state.best_score_full)
                best_variants = deepcopy(st.session_state.generated_variants)
                
                st.session_state.intermediate_saves = [] 
                
                status_placeholder = st.empty() 
                
                # ==========================================================
                # FAZA 1: Căutare Aleatorie (Random Search)
                # ==========================================================
                st.info(f"FAZA 1 (Aleatorie): Target WIN: **{target_win_score}**. Max Încercări: {MAX_RANDOM_ATTEMPTS:,}.")
                
                with st.spinner(f"Se caută eșantionul optim (Fitness, WIN, 3/3, 2/2)..."):
                    while attempts < MAX_RANDOM_ATTEMPTS:
                        attempts += 1
                        
                        indices = list(range(len(st.session_state.variants)))
                        random.shuffle(indices)
                        current_variants = [st.session_state.variants[i] for i in indices[:count]]
                        
                        current_score = calculate_wins(
                            current_variants, st.session_state.rounds, 
                            round_weights, use_deviation_penalty, penalty_factor_k
                        )
                        
                        # Salvare Intermediară
                        if attempts % INTERMEDIATE_SAVE_INTERVAL == 0 and attempts > 0:
                            st.session_state.intermediate_saves.append({
                                'attempt': attempts,
                                'score': best_score.copy(),
                                'variants': deepcopy(best_variants)
                            })
                            status_placeholder.markdown(f'<div class="status-box"><span class="intermediate-save">💾 Salvare intermediară la {attempts:,} încercări.</span></div>', unsafe_allow_html=True)
                            time.sleep(0.01)
                        
                        if compare_scores(current_score, best_score, target_win_score):
                            best_score = current_score
                            best_variants = deepcopy(current_variants)

                        if attempts % 50 == 0 or best_score['win_score'] >= target_win_score or attempts >= MAX_RANDOM_ATTEMPTS:
                            score_detail_html = (
                                f"FITNESS: **{best_score['fitness_score']:.0f}** | WINs: {best_score['win_score']:,} | 3/3: {best_score['score_3_3']:,} | 2/2: {best_score['score_2_2']:,} | StdDev: {best_score['std_dev_wins']:.2f}"
                            )
                            status_html = f"""
                            <div class="status-box">
                                FAZA 1 (ALEATORIE): Încercări: {attempts:,}/{MAX_RANDOM_ATTEMPTS:,}
                                <div class="score-detail">Cel Mai Bun Scor: {score_detail_html}</div>
                            </div>
                            """
                            status_placeholder.markdown(status_html, unsafe_allow_html=True)
                            time.sleep(0.01)
                        
                        if best_score['win_score'] >= target_win_score: break
                            
                    st.session_state.optimization_attempts = attempts
                    status_placeholder.empty()

                    # ==========================================================
                    # FAZA 2: Căutare Evolutivă Avansată (Hole Coverage)
                    # ==========================================================
                    
                    if best_variants and local_search_iterations > 0:
                         st.info(f"FAZA 2 (Evolutivă): Căutare Avansată (Hole Coverage) cu **{local_search_iterations:,}** iterații.")
                         
                         local_status_placeholder = st.empty()
                         current_best_variants = deepcopy(best_variants)
                         current_best_score = best_score.copy()
                         pool_variants = st.session_state.variants
                         
                         for local_attempts in range(1, local_search_iterations + 1):
                            # ... (Logica de optimizare evolutivă - Păstrată) ...
                            # Logica de selecție slabă, găuri, mutație
                            variant_scores = {}
                            for idx, variant_data in enumerate(current_best_variants):
                                score_single = calculate_wins([variant_data], st.session_state.rounds)
                                score_metric = score_single['win_score'] * 1000 + score_single['score_3_3'] * 100 + score_single['score_2_2']
                                variant_scores[idx] = score_metric
                                
                            variant_to_replace_index = min(variant_scores, key=variant_scores.get)
                            
                            weakest_rounds = sorted(
                                current_best_score['score_per_round'].items(),
                                key=lambda item: item[1]['wins'],
                                reverse=False
                            )[:NUM_WEAK_ROUNDS_FOR_HOLE_ANALYSIS]

                            weak_round_sets = [item[0] for item in weakest_rounds]
                            
                            best_hole_coverage_score = -1
                            new_variant_candidate = None

                            for variant in pool_variants:
                                if variant in current_best_variants: continue
                                    
                                variant_set = set(int(n) for n in variant['combination'].split() if n.isdigit())
                                
                                hole_coverage_score = 0
                                for runda_set in weak_round_sets:
                                    matches = len(variant_set.intersection(runda_set))
                                    if matches >= 4: hole_coverage_score += 1000
                                    elif matches == 3: hole_coverage_score += 100
                                    elif matches == 2: hole_coverage_score += 10

                                if hole_coverage_score > best_hole_coverage_score:
                                    best_hole_coverage_score = hole_coverage_score
                                    new_variant_candidate = variant
                            
                            if new_variant_candidate is not None:
                                new_variant = new_variant_candidate
                            else:
                                new_variant = pool_variants[random.choice(range(len(pool_variants)))]
                                
                            test_variants = current_best_variants.copy()
                            test_variants[variant_to_replace_index] = new_variant
                            
                            test_score = calculate_wins(
                                test_variants, st.session_state.rounds, 
                                round_weights, use_deviation_penalty, penalty_factor_k
                            )
                            
                            if compare_scores(test_score, current_best_score, target_win_score):
                                current_best_score = test_score.copy()
                                current_best_variants = test_variants.copy()
                                
                            # Actualizare status
                            if local_attempts % 250 == 0 or local_attempts == local_search_iterations:
                                score_detail_html = (
                                    f"FITNESS: **{current_best_score['fitness_score']:.0f}** | WINs: {current_best_score['win_score']:,} | 3/3: {current_best_score['score_3_3']:,} | 2/2: {current_best_score['score_2_2']:,} | StdDev: {current_best_score['std_dev_wins']:.2f}"
                                )
                                local_status_html = f"""
                                <div class="status-box local-search-status">
                                    FAZA 2 (EVOLUTIVĂ): Căutare Evolutivă: {local_attempts:,}/{local_search_iterations:,}
                                    <div class="score-detail">Cel Mai Bun Scor: {score_detail_html}</div>
                                </div>
                                """
                                local_status_placeholder.markdown(local_status_html, unsafe_allow_html=True)
                                time.sleep(0.01)

                         best_score = current_best_score
                         best_variants = current_best_variants.copy()
                         local_status_placeholder.empty()
                         st.success(f"FAZA 2 Finalizată. Scor îmbunătățit la **WIN {best_score['win_score']}** și **FITNESS {best_score['fitness_score']:.0f}** după {local_attempts:,} iterații evolutive.")
                    else:
                        local_attempts = 0

                    # Actualizarea Finală a Stării
                    st.session_state.local_search_attempts = local_attempts
                    st.session_state.generated_variants = best_variants
                    st.session_state.best_score_full = best_score
                    
                    performance_text = analyze_round_performance(best_variants, st.session_state.rounds)
                    st.session_state.round_performance_text = performance_text
                        
                    st.success(f"🏆 Optimizare finalizată! Fitness final: **{best_score['fitness_score']:.0f}**.")
                    st.balloons()
                    st.rerun() 
        
        if st.session_state.optimization_attempts > 0 or st.session_state.local_search_attempts > 0:
             st.info(f"Ultima rulare a folosit **{st.session_state.optimization_attempts:,}** încercări aleatorii și **{st.session_state.local_search_attempts:,}** încercări evolutive.")


# TAB 3: Rezultate & Analiză (Păstrată)
with tab3:
    st.markdown("## 📊 Rezultate și Analiză (Ultra Premium)")
    
    if not st.session_state.generated_variants:
        st.info("ℹ️ Nu există rezultate generate încă.")
    else:
        # --- Statistici de Scor ---
        col1, col2, col3, col4, col5 = st.columns(5)
        full_score = st.session_state.best_score_full
        
        with col1: st.metric("Fitness Final", f"{full_score['fitness_score']:.0f}")
        with col2: st.metric("Scor WIN (Brut)", full_score['win_score'])
        with col3: st.metric("Scor 3/3", f"{full_score['score_3_3']:,}")
        with col4: st.metric("Ecart Tip (Deviație)", f"{full_score['std_dev_wins']:.2f}")
        with col5: st.metric("Scor Ponderat", f"{full_score['weighted_score_sum']:.0f}")
        
        st.markdown("---")
        
        # --- Vizualizare Frecvență (Păstrată) ---
        col_list, col_chart = st.columns([1, 2])
        
        with col_list:
            st.markdown("### 📋 Lista de Variante")
            df_results = pd.DataFrame(st.session_state.generated_variants)
            st.download_button("💾 Descarcă TXT", data=variants_to_text(st.session_state.generated_variants), file_name=f"variante_optim_{len(st.session_state.generated_variants)}.txt", mime="text/plain", use_container_width=True)
            st.download_button("📊 Descarcă CSV", data=variants_to_csv(st.session_state.generated_variants), file_name=f"variante_optim_{len(st.session_state.generated_variants)}.csv", mime="text/csv", use_container_width=True)
            st.dataframe(df_results, use_container_width=True, hide_index=True, height=250)
        
        with col_chart:
            st.markdown("### 🔥 Analiza Frecvenței Numerelor (Heatmap)")
            df_freq = analyze_number_frequency(st.session_state.generated_variants)
            
            try:
                fig = px.bar(
                    df_freq,
                    x='Număr',
                    y='Frecvență',
                    text='Frecvență',
                    title='Frecvența de Apariție a Numerelor în Eșantionul Optimizat',
                    labels={'Număr': 'Număr (1-49)', 'Frecvență': 'Număr de Apariții'}
                )
                fig.update_traces(marker_color='#667eea', textposition='outside')
                fig.update_layout(xaxis={'tickmode': 'linear', 'dtick': 5}, yaxis={'tickformat': ','})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Eșantionul conține {len(df_results)} variante. Frecvența medie per număr ar trebui să fie de aproximativ {(len(df_results) * 6) / 49:.2f} apariții.")
            except NameError:
                 st.error("Eroare: Modulul Plotly nu a putut fi încărcat. Verificați `requirements.txt`.")
            except:
                 st.warning("Nu există suficiente date pentru a genera graficul de frecvență.")


        st.markdown("---")
        
        # --- Analiza Forței Variantelor în Pool-ul Mare ---
        st.markdown("### ⚡ Analiza Forței Variantelor în Pool-ul Mare (Calitatea materiei prime)")
        
        if st.session_state.variants and st.session_state.rounds:
            df_strength = analyze_variant_strength(st.session_state.variants, st.session_state.rounds)
            
            col_top, col_bottom = st.columns(2)
            
            with col_top:
                st.markdown("#### Top 10 Variante (Cele mai Bune individual)")
                st.dataframe(df_strength.head(10), use_container_width=True, hide_index=True)
            
            with col_bottom:
                st.markdown("#### Bottom 10 Variante (Cele mai Slabe individual)")
                st.dataframe(df_strength.tail(10), use_container_width=True, hide_index=True)
            
            st.caption("Acest clasament arată performanța istorică a fiecărei variante din pool-ul tău mare, fiind util pentru curățarea manuală a pool-ului de bază.")
        else:
            st.info("Încărcați variante și runde pentru a rula Analiza de Forță.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>👑 Generator Variante Loterie (Premium) | Algoritmi Evolutivi & Analiză Avansată</p>
    </div>
    """,
    unsafe_allow_html=True
)
