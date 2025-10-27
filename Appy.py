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
# CONSTANTE ȘI CONFIGURARE PAGINĂ (Păstrată)
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

# --- CSS NOU (Adăugat pentru culori specifice live-score-ului) ---
st.markdown("""
    <style>
    /* Stiluri generale păstrate */
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #667eea; }
    h1 { color: white !important; text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem; }
    h2, h3 { color: #667eea !important; }
    .status-box { 
        border: 2px solid #667eea; padding: 10px; border-radius: 8px; 
        background-color: #f0f2f6; color: #333333; font-size: 1.05rem; 
        font-weight: bold; text-align: center; margin-top: 10px; 
    }
    .score-detail { font-size: 0.9rem; font-weight: normal; margin-top: 5px; }
    .intermediate-save { color: #27ae60; font-size: 0.8rem; margin-top: 5px; }

    /* --- NOU: LIVE SCORE BOX STYLING --- */
    .live-score-container {
        display: flex;
        justify-content: space-around;
        gap: 10px;
        margin-top: 10px;
    }
    .live-score-box {
        flex: 1;
        padding: 8px 5px;
        border-radius: 6px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 1.2;
    }
    /* Culori pentru fiecare categorie */
    .win-4-plus { background-color: #e74c3c; border: 1px solid #c0392b; } /* Roșu închis */
    .score-3-3 { background-color: #f39c12; border: 1px solid #e67e22; } /* Portocaliu */
    .score-2-2 { background-color: #2ecc71; border: 1px solid #27ae60; } /* Verde deschis */

    .score-label { font-size: 0.7rem; opacity: 0.8; }
    .score-value { font-size: 1.2rem; }

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
if 'params' not in st.session_state:
    st.session_state.params = {
        'count': 1165,
        'target_wins_plus': 10,
        'local_search_iterations': 5000,
        'use_recency_weighting': True,
        'use_deviation_penalty': True,
        'penalty_factor_k': PENALTY_FACTOR_K
    }

# Funcțiile `calculate_wins`, `compare_scores`, `get_round_weights`, `analyze_variant_strength` etc. 
# sunt păstrate exact ca în versiunea anterioară. 
# (Omise pentru concizie în acest răspuns, dar prezente în codul real.)

# =========================================================================
# FUNCȚIA DE VIZUALIZARE LIVE A SCORULUI (NOU)
# =========================================================================

def create_live_score_html(score, current_attempts, total_attempts, phase_name):
    """Generează HTML pentru afișarea statusului de optimizare detaliat."""
    
    # 1. Status General (Mutat mai jos pentru a fi afișat deasupra scorurilor)
    status_html = f"""
    <div class="status-box">
        FAZA {phase_name}: Încercări: {current_attempts:,}/{total_attempts:,}
        <div class="score-detail">Fitness (Ponderat): **{score['fitness_score']:.0f}** | StdDev: {score['std_dev_wins']:.2f}</div>
    </div>
    """

    # 2. Status Detaliat pe Scorul Multi-Obiectiv
    score_html = f"""
    <div class="live-score-container">
        <div class="live-score-box win-4-plus">
            <div class="score-label">WIN (4+)</div>
            <div class="score-value">{score['win_score']:,}</div>
        </div>
        <div class="live-score-box score-3-3">
            <div class="score-label">3/3</div>
            <div class="score-value">{score['score_3_3']:,}</div>
        </div>
        <div class="live-score-box score-2-2">
            <div class="score-label">2/2</div>
            <div class="score-value">{score['score_2_2']:,}</div>
        </div>
    </div>
    """
    return status_html + score_html


# =========================================================================
# STREAMLIT UI & LOGIC FLOW (Modificată doar bucla de optimizare)
# =========================================================================

# Header, Sidebar, Tab 1 (Păstrate)
# ...

# Tab 2 (Modificată logica de afișare a statusului în timpul rulării)
with st.container(): # Folosim un container pentru a evita repetarea codului Tab 2
    
    # ... (Codul de configurare a parametrilor)
    
    if st.button("🚀 Generează & Optimizează (Evolutiv)", use_container_width=True, type="primary"):
        
        # ... (Logica de colectare parametri și inițializare - Păstrată)
        
        # Aici începe logica de optimizare (Modificată)
        if not st.session_state.rounds:
            st.error("Vă rugăm să încărcați sau să introduceți runde mai întâi.")
        else:
            
            # Colectare Parametri
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
            
            # Placeholder-ul va deține AMBELE casete: status general ȘI scoruri detaliate
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
                        # Afișăm separat notificarea de salvare intermediară
                        st.info(f"💾 Salvare intermediară la {attempts:,} încercări.")
                        time.sleep(0.01)
                    
                    if compare_scores(current_score, best_score, target_win_score):
                        best_score = current_score
                        best_variants = deepcopy(current_variants)

                    if attempts % 50 == 0 or best_score['win_score'] >= target_win_score or attempts >= MAX_RANDOM_ATTEMPTS:
                        # --- NOU: AFIȘARE LIVE SCORURI DETALIATE ---
                        status_html = create_live_score_html(
                            best_score, 
                            current_attempts=attempts, 
                            total_attempts=MAX_RANDOM_ATTEMPTS, 
                            phase_name="1 (ALEATORIE)"
                        )
                        status_placeholder.markdown(status_html, unsafe_allow_html=True)
                        time.sleep(0.01) # Pauză scurtă pentru a permite actualizarea UI-ului
                    
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
                         # ... (Logica de selecție slabă, găuri, mutație - Păstrată)
                         # ... (Calculează test_score)
                         
                         if local_attempts % 250 == 0 or local_attempts == local_search_iterations:
                            # --- NOU: AFIȘARE LIVE SCORURI DETALIATE ---
                            status_html = create_live_score_html(
                                current_best_score, 
                                current_attempts=local_attempts, 
                                total_attempts=local_search_iterations, 
                                phase_name="2 (EVOLUTIVĂ)"
                            )
                            local_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                            time.sleep(0.01)

                     best_score = current_best_score
                     best_variants = current_best_variants.copy()
                     local_status_placeholder.empty()
                     st.success(f"FAZA 2 Finalizată. Scor îmbunătățit la **WIN {best_score['win_score']}** și **FITNESS {best_score['fitness_score']:.0f}** după {local_attempts:,} iterații evolutive.")
                else:
                    local_attempts = 0

                # Actualizarea Finală a Stării (Păstrată)
                # ...
                
# Tab 3 (Păstrată)
# ...

# Footer (Păstrat)
# ...

