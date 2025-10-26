import streamlit as st
import pandas as pd
import random
from io import StringIO
import time
from copy import deepcopy

# =========================================================================
# CONFIGURARE PAGINĂ ȘI CSS
# =========================================================================

# Configurare pagină
st.set_page_config(
    page_title="Generator Variante Loterie",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizat
st.markdown("""
    <style>
    /* Fundalul general al aplicației */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    /* Dimensiunea și culoarea metricilor */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #667eea;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    /* Titluri H1 */
    h1 {
        color: white !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    /* Titluri H2, H3 */
    h2, h3 {
        color: #667eea !important;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    /* Stil pentru chenarul cu rezultate (Performanță Rundă) */
    .results-box {
        border: 1px solid #764ba2;
        padding: 15px;
        border-radius: 8px;
        background-color: #333333; 
        color: white; 
        height: 400px; 
        overflow-y: scroll;
        font-family: monospace; 
    }
    .results-box p {
        color: white; 
        margin: 5px 0;
    }
    /* Stil pentru chenarul de status compact în timpul optimizării */
    .status-box {
        border: 2px solid #667eea;
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6; 
        color: #333333;
        font-size: 1.05rem;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    /* Stil pentru statusul de căutare locală */
    .local-search-status {
        color: #e67e22; 
    }
    /* Scorul complet afisat in chenarul de status */
    .score-detail {
        font-size: 0.9rem;
        font-weight: normal;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# INITIALIZARE SESIUNE ȘI FUNCȚII UTILITY
# =========================================================================

# Setează o limită de siguranță pentru căutarea aleatorie
MAX_RANDOM_ATTEMPTS = 100000

# Inițializare session state
if 'variants' not in st.session_state:
    st.session_state.variants = []
if 'generated_variants' not in st.session_state:
    st.session_state.generated_variants = []
if 'internal_duplicates' not in st.session_state:
    st.session_state.internal_duplicates = 0
if 'inter_duplicates' not in st.session_state:
    st.session_state.inter_duplicates = 0
if 'rounds' not in st.session_state:
    st.session_state.rounds = []
if 'rounds_raw' not in st.session_state:
    st.session_state.rounds_raw = []
if 'win_score' not in st.session_state:
    st.session_state.win_score = 0
if 'best_score_full' not in st.session_state:
     # NOU: Stochează scorul multi-obiectiv complet
    st.session_state.best_score_full = {'win_score': 0, 'score_3_3': 0, 'score_2_2': 0}
if 'round_performance_text' not in st.session_state:
    st.session_state.round_performance_text = ""
if 'manual_rounds_input' not in st.session_state:
    st.session_state.manual_rounds_input = ""
if 'optimization_attempts' not in st.session_state:
    st.session_state.optimization_attempts = 0
if 'local_search_attempts' not in st.session_state:
    st.session_state.local_search_attempts = 0


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
        if not line:
            continue
        parts = line.split(',', 1)
        if len(parts) != 2:
            errors.append(f"Linia {i}: Format invalid (lipsește virgula)")
            continue
        variant_id = parts[0].strip()
        numbers = parts[1].strip()
        
        if not variant_id:
            errors.append(f"Linia {i}: ID lipsă")
            continue
        if not numbers:
            errors.append(f"Linia {i}: Combinație lipsă")
            continue
        
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

def process_round_text(text):
    """Funcție utilitară pentru a procesa textul rundelor (din fișier sau manual)."""
    rounds_set_list = []
    rounds_display_list = []
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Extrage numerele
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
    if rounds_file is None:
        return [], []
    
    try:
        content = rounds_file.getvalue().decode("utf-8")
        return process_round_text(content)
    except Exception as e:
        st.error(f"Eroare la procesarea fișierului de runde: {e}")
        return [], []


def calculate_wins(generated_variants, rounds):
    """
    Calculează scorul multi-obiectiv (WIN (>=4/4), 3/3 și 2/2)
    pentru un set de variante față de un set de extrageri (rounds).
    """
    if not rounds or not generated_variants:
        return {'win_score': 0, 'score_3_3': 0, 'score_2_2': 0}
    
    total_wins = 0 # >= 4/4
    total_3_3 = 0
    total_2_2 = 0
    
    # Pre-procesează setul de variante o singură dată
    variant_sets = []
    for variant_data in generated_variants:
        try:
            variant_sets.append(set(int(n) for n in variant_data['combination'].split() if n.isdigit()))
        except:
            continue
    
    for variant_set in variant_sets:
        # Pentru fiecare runda (extragere)
        for runda_set in rounds:
            # Calculează intersecția (potrivirile)
            matches = len(variant_set.intersection(runda_set))
            
            if matches >= 4:
                # Tratează ca WIN
                total_wins += 1
            elif matches == 3:
                # Tratează ca 3/3
                total_3_3 += 1
            elif matches == 2:
                # Tratează ca 2/2
                total_2_2 += 1
                
    return {
        'win_score': total_wins,
        'score_3_3': total_3_3,
        'score_2_2': total_2_2
    }


def compare_scores(current_score, best_score, target_win_score):
    """
    Compară două scoruri multi-obiectiv folosind ierarhia de priorități:
    1. WIN-uri (>= 4/4) - Prioritate 1 (P1)
    2. Scor 3/3 - P2
    3. Scor 2/2 - P3
    Returnează True dacă current_score este mai bun.
    """
    
    # P1: Verifică dacă scorul WIN atinge sau depășește ținta
    if current_score['win_score'] >= target_win_score and best_score['win_score'] < target_win_score:
        return True
    
    # P1: Compară scorul WIN (pentru a găsi cel mai bun în caz de non-atingere a țintei)
    if current_score['win_score'] > best_score['win_score']:
        return True
    if current_score['win_score'] < best_score['win_score']:
        return False
        
    # P2: Dacă WIN-urile sunt egale, compară 3/3
    if current_score['score_3_3'] > best_score['score_3_3']:
        return True
    if current_score['score_3_3'] < best_score['score_3_3']:
        return False
        
    # P3: Dacă și 3/3 sunt egale, compară 2/2
    if current_score['score_2_2'] > best_score['score_2_2']:
        return True
        
    return False

def analyze_round_performance(generated_variants, rounds_set):
    """
    Calculează performanța pe rundă și returnează un string mare,
    formatat: "Runda X - Y variante câștigătoare".
    """
    if not rounds_set or not generated_variants:
        return ""

    variant_sets = []
    for variant_data in generated_variants:
        try:
            variant_sets.append(set(int(n) for n in variant_data['combination'].split() if n.isdigit()))
        except:
            continue

    if not variant_sets:
         return ""

    results_lines = []
    for i, runda_set in enumerate(rounds_set):
        wins_in_round = 0
        score_3_3_in_round = 0
        score_2_2_in_round = 0
        
        for v_set in variant_sets:
            matches = len(v_set.intersection(runda_set))
            
            if matches >= 4:
                wins_in_round += 1
            elif matches == 3:
                score_3_3_in_round += 1
            elif matches == 2:
                score_2_2_in_round += 1
        
        results_lines.append(f"Runda {i+1} - WINs: {wins_in_round}, 3/3: {score_3_3_in_round}, 2/2: {score_2_2_in_round}")
        
    return '\n'.join(results_lines)


def generate_sample_data(count=100):
    """Generează date de exemplu, incluzând DUPLICATE PENTRU TESTARE"""
    sample_data = []
    sample_data.append(f"1, 5 7 44 32 18")
    sample_data.append(f"2, 12 23 34 34 49")
    sample_data.append(f"3, 7 5 44 32 18") 
    for i in range(4, count + 1):
        numbers = [str(random.randint(1, 49)) for _ in range(6)]
        sample_data.append(f"{i}, {' '.join(numbers)}")
    return '\n'.join(sample_data)

def variants_to_text(variants):
    """Convertește variantele în text (ID, numere separate prin spațiu)"""
    return '\n'.join([f"{v['id']},{v['combination']}" for v in variants])

def variants_to_csv(variants):
    """Convertește variantele în CSV"""
    df = pd.DataFrame(variants)
    return df.to_csv(index=False)

# =========================================================================
# STREAMLIT UI & LOGIC FLOW
# =========================================================================

# Header
st.markdown("# 🎲 Generator Variante Loterie")
st.markdown("### Gestionează și generează variante aleatorii pentru loterie")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Statistici")
    st.metric("Variante Curățate", len(st.session_state.variants))
    st.metric("Variante Generate", len(st.session_state.generated_variants))
    st.metric("Runde Încărcate", len(st.session_state.rounds_raw))
    st.markdown("---")
    st.markdown("#### Scor Obținut")
    
    full_score = st.session_state.best_score_full
    st.metric(
        "WIN (≥4/4)",
        full_score['win_score']
    )
    st.caption(f"3/3: {full_score['score_3_3']:,} | 2/2: {full_score['score_2_2']:,}")
    
    st.markdown("---")
    st.markdown("#### Optimizare")
    if st.session_state.optimization_attempts > 0:
         st.metric("Încercări Aleatorii", f"{st.session_state.optimization_attempts:,}")
    if st.session_state.local_search_attempts > 0:
         st.metric("Încercări Locale", f"{st.session_state.local_search_attempts:,}")

    st.markdown("---")
    st.markdown("## 🧹 Duplicate Eliminate")
    st.metric("În Combinații (Interne)", st.session_state.internal_duplicates)
    st.metric("Între Combinații (Inter-Variante)", st.session_state.inter_duplicates)

    st.markdown("---")
    st.markdown("## ℹ️ Informații")
    st.info("Algoritmul optimizează după WIN (>=4/4), apoi 3/3, apoi 2/2.")
    
    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.variants = []
        st.session_state.generated_variants = []
        st.session_state.internal_duplicates = 0
        st.session_state.inter_duplicates = 0
        st.session_state.rounds = []
        st.session_state.rounds_raw = []
        st.session_state.win_score = 0
        st.session_state.best_score_full = {'win_score': 0, 'score_3_3': 0, 'score_2_2': 0}
        st.session_state.round_performance_text = ""
        st.session_state.manual_rounds_input = ""
        st.session_state.optimization_attempts = 0
        st.session_state.local_search_attempts = 0
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante & Curăță", "🎲 Generează Random & Calculează Win", "📊 Rezultate"])

# TAB 1: Încărcare Variante
with tab1:
    # ... (Codul pentru Tab 1 rămâne neschimbat)
    st.markdown("## 📝 Pas 1: Încarcă Variantele Tale & Curăță Duplicatele")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Introdu variantele")
        st.caption("Format: ID, numere separate prin spațiu (ex: 1, 5 7 44 32 18)")
    
    with col2:
        if st.button("✨ Generează Date Exemplu", use_container_width=True):
            sample = generate_sample_data(100)
            st.session_state.sample_data = sample
            st.success("✅ S-au generat 100 variante exemplu (inclusiv duplicate pentru testare)!")
    
    # Textarea pentru input
    default_value = st.session_state.get('sample_data', '')
    variants_input = st.text_area(
        "Variante",
        value=default_value,
        height=300,
        placeholder="Exemplu:\n1, 5 7 44 32 18\n2, 12 23 34 34 49\n3, 7 5 44 32 18",
        label_visibility="collapsed"
    )
    
    # Butoane de acțiune
    col_load, col_file, col_download_baza = st.columns([2, 2, 2])
    
    with col_load:
        if st.button("📥 Încarcă & Curăță Variante", use_container_width=True, type="primary"):
            if not variants_input.strip():
                st.error("❌ Te rog să introduci variante!")
            else:
                with st.spinner("Se încarcă și se curăță variantele..."):
                    
                    variants, errors, internal_duplicates, inter_duplicates = parse_variants(variants_input)
                    
                    st.session_state.variants = variants
                    st.session_state.internal_duplicates = internal_duplicates
                    st.session_state.inter_duplicates = inter_duplicates
                    st.session_state.sample_data = variants_input
                    
                    if variants:
                        st.success(f"✅ S-au încărcat {len(variants)} variante unice cu succes!")
                        st.info(f"S-au eliminat {internal_duplicates} numere duplicate din combinații și {inter_duplicates} variante complet identice.")
                        
                        if errors:
                            with st.expander("⚠️ Avertismente"):
                                for error in errors:
                                    st.warning(error)
                    else:
                        st.error("❌ Nu s-au putut încărca variante valide!")
                        if errors:
                            for error in errors:
                                st.error(error)
    
    with col_file:
        uploaded_file = st.file_uploader(
            "Sau încarcă fișier TXT/CSV",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            variants, errors, internal_duplicates, inter_duplicates = parse_variants(content)
            
            st.session_state.variants = variants
            st.session_state.internal_duplicates = internal_duplicates
            st.session_state.inter_duplicates = inter_duplicates
            
            if variants:
                st.success(f"✅ S-au încărcat {len(variants)} variante unice din fișier!")
            else:
                st.error("❌ Fișierul nu conține variante valide!")
    
    with col_download_baza:
        if st.session_state.variants:
            st.download_button(
                "💾 Descarcă Variante Curățate",
                data=variants_to_text(st.session_state.variants),
                file_name="variante_curatate_unice.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    if st.session_state.variants:
        st.markdown("---")
        st.markdown("### 👀 Previzualizare Variante Curățate")
        
        df_preview = pd.DataFrame(st.session_state.variants)
        
        if len(st.session_state.variants) > 10:
            st.dataframe(df_preview.head(5), use_container_width=True, hide_index=True)
            st.dataframe(df_preview.tail(5), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
# END TAB 1

# TAB 2: Generare Random & Calcul WIN
with tab2:
    st.markdown("## 🎲 Pas 2: Generează Variante Random & Calculează Performanța")
    
    if not st.session_state.variants:
        st.warning("⚠️ Nu există variante curățate încă! Mergi la tab-ul 'Încarcă Variante & Curăță'.")
        
    else:
        # -------------------------------------------------------------------------
        # Secțiunea 1: Încărcare Runde
        # -------------------------------------------------------------------------
        st.markdown("### 1. Încarcă Rundele (Extragerile) de Bază")
        
        col_file, col_manual = st.columns(2)
        
        # Opțiunea 1: Încărcare din fișier
        rounds_file = col_file.file_uploader(
            "Încărcați fișierul cu Rundele (extragerile)",
            type=['txt', 'csv'],
            key="rounds_uploader"
        )

        # Opțiunea 2: Adăugare manuală
        manual_rounds_input = col_manual.text_area(
            "Sau adaugă runde manual (câte o rundă pe linie, numere separate prin spațiu sau virgulă)",
            value=st.session_state.manual_rounds_input,
            height=100,
            placeholder="Exemplu:\n1 5 7 12 44 49\n2 10 20 30 40 45"
        )
        st.session_state.manual_rounds_input = manual_rounds_input
        
        # Procesarea și combinarea rundelor
        rounds_from_file_set, rounds_from_file_raw = parse_rounds_file(rounds_file)
        rounds_from_manual_set, rounds_from_manual_raw = process_round_text(manual_rounds_input)

        all_rounds_set_dict = {}
        for r_set, r_raw in zip(rounds_from_file_set, rounds_from_file_raw):
            all_rounds_set_dict[frozenset(r_set)] = r_raw
        for r_set, r_raw in zip(rounds_from_manual_set, rounds_from_manual_raw):
            all_rounds_set_dict[frozenset(r_set)] = r_raw
            
        st.session_state.rounds = list(all_rounds_set_dict.keys())
        st.session_state.rounds_raw = list(all_rounds_set_dict.values())
        
        st.metric("Total Runde Unice Încărcate", len(st.session_state.rounds))
        
        
        # -------------------------------------------------------------------------
        # Secțiunea 2: Previzualizare Runde ȘI Performanță
        # -------------------------------------------------------------------------
        if st.session_state.rounds_raw and st.session_state.round_performance_text:
            
            st.markdown("#### 🎯 Performanța Eșantionului pe Rundă (WINs, 3/3, 2/2)")
            
            performance_html = '<br>'.join([f"<p>{line}</p>" for line in st.session_state.round_performance_text.split('\n')])
            
            st.markdown(
                f'<div class="results-box">{performance_html}</div>',
                unsafe_allow_html=True
            )
            st.markdown("---")
        elif st.session_state.rounds_raw and not st.session_state.generated_variants:
             st.info("Încarcă și Generează Variante (Pasul 2) pentru a vedea Performanța pe Rundă.")
        
        # -------------------------------------------------------------------------
        # Secțiunea 3: Generare Random & Calcul
        # -------------------------------------------------------------------------
        st.markdown("### 2. Generare Eșantion Aleatoriu & Calcul Score")

        col_count, col_mode = st.columns([1, 1])
        
        with col_count:
            st.markdown(f"Ai **{len(st.session_state.variants)}** variante unice disponibile.")
            
            count = st.number_input(
                "Câte variante să generez pe eșantion?",
                min_value=1,
                max_value=len(st.session_state.variants),
                value=min(1165, len(st.session_state.variants)),
                step=1
            )
        
        with col_mode:
            st.markdown("#### Mod Generare")
            
            # Opțiunea de Optimizare
            optimize_mode = st.checkbox(
                "Mod Generare Optimă (Multi-Obiectiv) + Local Search",
                help="Rulează automat generarea random până atinge scorul WIN țintă sau limita de încercări, apoi rafinează cu Local Search."
            )
            
            target_wins_plus = st.number_input(
                "WINs (>=4/4) Țintă Suplimentare (+X)",
                min_value=0,
                value=10,
                step=1,
                disabled=not optimize_mode,
                help="Scorul țintă va fi: (Runde totale) + X"
            )
            
            local_search_iterations = st.number_input(
                "Iterații Căutare Locală",
                min_value=0,
                value=5000, 
                step=100,
                disabled=not optimize_mode,
                help="Numărul de încercări de rafinare (schimbare 1 variantă) după găsirea celui mai bun eșantion."
            )


        
        col_button, col_metric = st.columns(2)
        
        with col_metric:
            st.markdown("### ")
            # Afișarea scorului principal
            st.metric("Scor WIN (Principal)", st.session_state.best_score_full['win_score'])

        with col_button:
            st.markdown("### ")
            st.markdown("### ")
            if st.button("🎲 Generează & Calculează", use_container_width=True, type="primary"):
                
                if not st.session_state.rounds:
                    st.error("Vă rugăm să încărcați sau să introduceți runde mai întâi.")
                    st.session_state.optimization_attempts = 0
                    st.session_state.local_search_attempts = 0
                
                else:
                    total_rounds = len(st.session_state.rounds)
                    target_win_score = total_rounds + target_wins_plus
                    attempts = 0
                    local_attempts = 0
                    
                    # Initializare best_score
                    best_score = {'win_score': -1, 'score_3_3': -1, 'score_2_2': -1}
                    best_variants = []
                    
                    # Reîncărcăm scorul precedent ca punct de plecare dacă există
                    if st.session_state.generated_variants:
                        best_score = st.session_state.best_score_full.copy()
                        best_variants = deepcopy(st.session_state.generated_variants)
                    
                    
                    # NOU: Placeholder pentru afișarea statusului compact
                    status_placeholder = st.empty() 
                    
                    if optimize_mode:
                        
                        # ==========================================================
                        # FAZA 1: Căutare Aleatorie (Random Search)
                        # ==========================================================
                        st.info(f"FAZA 1 (Aleatorie): Target WIN (>=4/4): **{target_win_score}**. Max Încercări: {MAX_RANDOM_ATTEMPTS:,}.")
                        
                        
                        with st.spinner(f"Se caută eșantionul optim (WIN + 3/3 + 2/2)..."):
                            
                            while best_score['win_score'] < target_win_score and attempts < MAX_RANDOM_ATTEMPTS:
                                attempts += 1
                                
                                # Extragere random
                                indices = list(range(len(st.session_state.variants)))
                                random.shuffle(indices)
                                current_variants = [st.session_state.variants[i] for i in indices[:count]]
                                
                                # Calcul scor multi-obiectiv
                                current_score = calculate_wins(current_variants, st.session_state.rounds)
                                
                                # Verificare și actualizare cel mai bun scor (folosind ierarhia de priorități)
                                if compare_scores(current_score, best_score, target_win_score):
                                    best_score = current_score
                                    best_variants = deepcopy(current_variants)

                                # Actualizare chenar la fiecare 50 de încercări
                                if attempts % 50 == 0 or best_score['win_score'] >= target_win_score or attempts >= MAX_RANDOM_ATTEMPTS:
                                    
                                    score_detail_html = (
                                        f"WINs: **{best_score['win_score']:,}** | "
                                        f"3/3: {best_score['score_3_3']:,} | "
                                        f"2/2: {best_score['score_2_2']:,}"
                                    )
                                    status_html = f"""
                                    <div class="status-box">
                                        Încercări Aleatorii: {attempts:,}/{MAX_RANDOM_ATTEMPTS:,}
                                        <div class="score-detail">Cel Mai Bun Scor: {score_detail_html}</div>
                                    </div>
                                    """
                                    status_placeholder.markdown(status_html, unsafe_allow_html=True)
                                    time.sleep(0.01)
                                
                                if not st.session_state.variants:
                                    break
                                
                        
                        st.session_state.optimization_attempts = attempts
                        
                        # Curățăm placeholder-ul și afișăm mesajul de încheiere a fazei
                        status_placeholder.empty()
                        
                        if best_score['win_score'] >= target_win_score:
                             st.success(f"FAZA 1 (Aleatorie) finalizată: Targetul WIN ({target_win_score}) atins după {attempts:,} încercări.")
                        elif attempts >= MAX_RANDOM_ATTEMPTS:
                             st.warning(f"FAZA 1 (Aleatorie) finalizată: Atingere limită ({MAX_RANDOM_ATTEMPTS:,} încercări). Trecere la rafinare...")

                        # ==========================================================
                        # FAZA 2: Căutare Locală (Local Search)
                        # ==========================================================
                        
                        # Rulăm Local Search doar dacă am găsit un eșantion valid și avem iterații setate
                        if best_variants and local_search_iterations > 0 and best_score['win_score'] < target_win_score * 2: # Evităm Local Search dacă WINs sunt deja mult prea mari
                             
                             st.info(f"FAZA 2 (Local Search): Rafinare scor existent ({best_score['win_score']} WINs) cu **{local_search_iterations:,}** iterații locale.")
                             
                             # Placeholder pentru statusul Local Search
                             local_status_placeholder = st.empty()

                             current_best_variants = deepcopy(best_variants)
                             current_best_score = best_score.copy()
                             
                             # Indexurile variantelor din pool-ul mare
                             pool_indices = list(range(len(st.session_state.variants)))
                             
                             for local_attempts in range(1, local_search_iterations + 1):
                                 
                                 # 1. Alege o variantă random din Eșantionul Curent de înlocuit (indexul in eșantion)
                                 variant_to_replace_index = random.randrange(count)
                                 
                                 # 2. Alege o variantă random din Pool-ul Mare (indexul in pool-ul mare)
                                 new_variant_pool_index = random.choice(pool_indices)
                                 new_variant = st.session_state.variants[new_variant_pool_index]
                                 
                                 # 3. Crează eșantionul test (schimbare "locală")
                                 test_variants = current_best_variants.copy()
                                 test_variants[variant_to_replace_index] = new_variant
                                 
                                 # 4. Calculează scorul
                                 test_score = calculate_wins(test_variants, st.session_state.rounds)
                                 
                                 # 5. Dacă e mai bun, înlocuiește eșantionul cel mai bun
                                 if compare_scores(test_score, current_best_score, target_win_score):
                                     current_best_score = test_score.copy()
                                     current_best_variants = test_variants.copy()
                                     
                                 # Actualizare status (mai rar, de exemplu la 250 de iterații)
                                 if local_attempts % 250 == 0 or local_attempts == local_search_iterations:
                                     score_detail_html = (
                                        f"WINs: **{current_best_score['win_score']:,}** | "
                                        f"3/3: {current_best_score['score_3_3']:,} | "
                                        f"2/2: {current_best_score['score_2_2']:,}"
                                    )
                                     local_status_html = f"""
                                     <div class="status-box local-search-status">
                                         Local Search: {local_attempts:,}/{local_search_iterations:,}
                                         <div class="score-detail">Cel Mai Bun Scor: {score_detail_html}</div>
                                     </div>
                                     """
                                     local_status_placeholder.markdown(local_status_html, unsafe_allow_html=True)
                                     time.sleep(0.01)

                             # Actualizează rezultatele finale cu Local Search
                             best_score = current_best_score
                             best_variants = current_best_variants.copy()
                             
                             # Curățăm placeholder-ul de status Local Search
                             local_status_placeholder.empty()
                             
                             st.success(f"FAZA 2 Finalizată. Scor îmbunătățit la **{best_score['win_score']} WINs** după {local_attempts:,} iterații locale.")
                        else:
                            local_attempts = 0 # Nu a rulat

                        
                        # Rezultatul final al optimizării
                        st.session_state.local_search_attempts = local_attempts
                        generated_variants = best_variants
                        
                        win_score = best_score['win_score']
                        win_message = f"🏆 Optimizare finalizată! Scor WIN (>=4/4): **{win_score}**."
                        
                    else:
                        # Generare Simplă (O singură rulare)
                        with st.spinner(f"Se generează {count} variante random și se calculează scorul..."):
                            
                            indices = list(range(len(st.session_state.variants)))
                            random.shuffle(indices)
                            generated_variants = [st.session_state.variants[i] for i in indices[:count]]
                            
                            best_score = calculate_wins(generated_variants, st.session_state.rounds)
                            win_score = best_score['win_score']
                            
                            st.session_state.optimization_attempts = 0 
                            st.session_state.local_search_attempts = 0 
                            win_message = f"✅ S-au generat {len(generated_variants)} variante. WIN: **{win_score}**."


                    # Actualizarea Session State
                    st.session_state.generated_variants = generated_variants
                    st.session_state.win_score = win_score
                    st.session_state.best_score_full = best_score
                    
                    performance_text = analyze_round_performance(generated_variants, st.session_state.rounds)
                    st.session_state.round_performance_text = performance_text
                        
                    st.success(win_message)
                    st.balloons()
                    
                    # Forțează o re-rulare completă a paginii doar la final, pentru a actualiza tab-urile
                    st.rerun() 
        
        # -------------------------------------------------------------------------
        # Secțiunea 4: Afișare Contor Încercări (dacă a rulat optimizarea)
        # -------------------------------------------------------------------------
        if st.session_state.optimization_attempts > 0 or st.session_state.local_search_attempts > 0:
             st.info(f"Ultima rulare a folosit **{st.session_state.optimization_attempts:,}** încercări aleatorii și **{st.session_state.local_search_attempts:,}** încercări locale.")


# TAB 3: Rezultate
with tab3:
    st.markdown("## 📊 Rezultate Generate")
    
    if not st.session_state.generated_variants:
        st.info("ℹ️ Nu există rezultate generate încă. Mergi la tab-ul 'Generează Random & Calculează Win' pentru a genera variante.")
    else:
        # Statistici
        col1, col2, col3, col4 = st.columns(4)
        full_score = st.session_state.best_score_full
        
        with col1:
            st.metric("Variante Generate", len(st.session_state.generated_variants))
        
        with col2:
            st.metric("Runde Folosite", len(st.session_state.rounds_raw))

        with col3:
            st.metric("Scor WIN (>=4/4)", full_score['win_score'])
            st.caption(f"Din total: {len(st.session_state.variants):,}")

        with col4:
            st.metric("Scor 3/3", f"{full_score['score_3_3']:,}")
            st.caption(f"Scor 2/2: {full_score['score_2_2']:,}")
        
        st.markdown("---")
        
        # Butoane de export
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "💾 Descarcă TXT",
                data=variants_to_text(st.session_state.generated_variants),
                file_name=f"variante_random_{len(st.session_state.generated_variants)}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "📊 Descarcă CSV",
                data=variants_to_csv(st.session_state.generated_variants),
                file_name=f"variante_random_{len(st.session_state.generated_variants)}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Generează Din Nou (Revino la Tab 2)", use_container_width=True):
                st.session_state.generated_variants = []
                st.rerun()
        
        st.markdown("---")
        
        # Afișare rezultate
        st.markdown("### 📋 Lista Completă de Variante Generate")
        
        df_results = pd.DataFrame(st.session_state.generated_variants)
        
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        with st.expander("🔍 Caută în rezultate"):
            search_term = st.text_input("Caută după ID sau combinație în rezultate")
            
            if search_term:
                filtered = [
                    v for v in st.session_state.generated_variants
                    if search_term.lower() in v['id'].lower() or 
                       search_term.lower() in v['combination'].lower()
                ]
                
                if filtered:
                    st.success(f"✅ S-au găsit {len(filtered)} rezultate")
                    st.dataframe(
                        pd.DataFrame(filtered),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("❌ Nu s-au găsit rezultate")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>🎲 Generator Variante Loterie | Creat cu ❤️ folosind Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
