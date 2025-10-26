import streamlit as st
import pandas as pd
import random
from io import StringIO
import time

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
    /* Stil pentru chenarul cu rezultate */
    .results-box {
        border: 1px solid #764ba2;
        padding: 15px;
        border-radius: 8px;
        /* Asigură contrastul: fundal întunecat, text deschis */
        background-color: #333333; 
        color: white; 
        height: 400px; 
        overflow-y: scroll;
        font-family: monospace; /* Font mono pentru lizibilitate */
    }
    .results-box p {
        color: white; /* Asigură că paragrafele din chenar sunt albe */
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# INITIALIZARE SESIUNE ȘI FUNCȚII UTILITY
# =========================================================================

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
if 'round_performance_text' not in st.session_state:
    st.session_state.round_performance_text = ""
if 'manual_rounds_input' not in st.session_state:
    st.session_state.manual_rounds_input = ""
if 'optimization_attempts' not in st.session_state: # NOU: Contor de încercări pentru optimizare
    st.session_state.optimization_attempts = 0


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
    """Calculează numărul total de potriviri."""
    if not rounds or not generated_variants:
        return 0
    
    total_wins = 0
    
    for variant_data in generated_variants:
        try:
            variant_set = set(int(n) for n in variant_data['combination'].split() if n.isdigit())
        except:
            continue
        
        for runda in rounds:
            # Dacă toate numerele din varianta generată (set) sunt un subset al rundei (set)
            if variant_set.issubset(runda):
                total_wins += 1
                
    return total_wins

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
        for v_set in variant_sets:
            if v_set.issubset(runda_set):
                wins_in_round += 1
        
        results_lines.append(f"Runda {i+1} - {wins_in_round} variante câștigătoare")
        
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
    st.metric("Scor Win Total", st.session_state.win_score)
    if st.session_state.optimization_attempts > 0:
         st.metric("Încercări Optimizare", st.session_state.optimization_attempts)
    st.markdown("---")
    st.markdown("## 🧹 Duplicate Eliminate")
    st.metric("În Combinații (Interne)", st.session_state.internal_duplicates)
    st.metric("Între Combinații (Inter-Variante)", st.session_state.inter_duplicates)

    st.markdown("---")
    st.markdown("## ℹ️ Informații")
    st.info("Aplicația elimină automat duplicatele și afișează scorul WIN pe baza rundelor încărcate.")
    
    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.variants = []
        st.session_state.generated_variants = []
        st.session_state.internal_duplicates = 0
        st.session_state.inter_duplicates = 0
        st.session_state.rounds = []
        st.session_state.rounds_raw = []
        st.session_state.win_score = 0
        st.session_state.round_performance_text = ""
        st.session_state.manual_rounds_input = ""
        st.session_state.optimization_attempts = 0
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante & Curăță", "🎲 Generează Random & Calculează Win", "📊 Rezultate"])

# TAB 1: Încărcare Variante
with tab1:
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
            
            st.markdown("#### 🎯 Performanța Eșantionului pe Rundă")
            
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
            
            # Opțiunea NOUĂ de Optimizare
            optimize_mode = st.checkbox(
                "Mod Generare Optimă (Targeted WINs)",
                help="Rulează automat generarea random până atinge scorul WIN țintă."
            )
            
            target_wins_plus = st.number_input(
                "WINs Țintă Suplimentare (+X)",
                min_value=0,
                value=10,
                step=1,
                disabled=not optimize_mode,
                help="Scorul țintă va fi: (Runde totale) + X"
            )

        
        col_button, col_metric = st.columns(2)
        
        with col_metric:
            st.markdown("### ")
            st.metric("Scor de Performanță Total", st.session_state.win_score)

        with col_button:
            st.markdown("### ")
            st.markdown("### ")
            if st.button("🎲 Generează & Calculează", use_container_width=True, type="primary"):
                
                if not st.session_state.rounds:
                    st.error("Vă rugăm să încărcați sau să introduceți runde mai întâi.")
                    st.session_state.optimization_attempts = 0
                
                else:
                    total_rounds = len(st.session_state.rounds)
                    target_win_score = total_rounds + target_wins_plus
                    attempts = 0
                    best_score = -1
                    best_variants = []
                    
                    if optimize_mode:
                        st.info(f"Target stabilit: {total_rounds} Runde + {target_wins_plus} WINs = **{target_win_score} WINs**.")
                        
                        # Generare Optimă (Buclă WHILE)
                        with st.spinner(f"Se caută eșantionul cu cel puțin {target_win_score} WINs..."):
                            
                            while best_score < target_win_score:
                                attempts += 1
                                
                                # Simulare extragere random
                                indices = list(range(len(st.session_state.variants)))
                                random.shuffle(indices)
                                current_variants = [st.session_state.variants[i] for i in indices[:count]]
                                
                                current_score = calculate_wins(current_variants, st.session_state.rounds)
                                
                                if current_score > best_score:
                                    best_score = current_score
                                    best_variants = current_variants
                                
                                # Feedback în timpul buclei
                                if attempts % 50 == 0:
                                    st.caption(f"Încercări: {attempts}. Cel mai bun scor până acum: {best_score} WINs.")
                                
                                if attempts > 5000: # Limitare pentru a evita blocarea aplicației în cazuri improbabile
                                    st.warning(f"Atenție: S-au efectuat {attempts} încercări fără a atinge ținta de {target_win_score} WINs. Se oprește la cel mai bun rezultat obținut ({best_score} WINs).")
                                    break
                        
                        st.session_state.optimization_attempts = attempts
                        generated_variants = best_variants
                        win_score = best_score
                        win_message = f"🏆 Optimizare reușită după **{attempts}** încercări! S-au obținut **{win_score} WINs** (Ținta: {target_win_score})."
                        
                    else:
                        # Generare Simplă (O singură rulare)
                        with st.spinner(f"Se generează {count} variante random și se calculează scorul..."):
                            
                            indices = list(range(len(st.session_state.variants)))
                            random.shuffle(indices)
                            generated_variants = [st.session_state.variants[i] for i in indices[:count]]
                            
                            win_score = calculate_wins(generated_variants, st.session_state.rounds)
                            st.session_state.optimization_attempts = 0 
                            win_message = f"✅ S-au generat {len(generated_variants)} variante și s-au obținut **{win_score} WINs**!"


                    # Actualizarea Session State
                    st.session_state.generated_variants = generated_variants
                    st.session_state.win_score = win_score
                    
                    performance_text = analyze_round_performance(generated_variants, st.session_state.rounds)
                    st.session_state.round_performance_text = performance_text
                        
                    st.success(win_message)
                    st.balloons()
                    st.rerun() 
        
        # -------------------------------------------------------------------------
        # Secțiunea 4: Afișare Contor Încercări (dacă a rulat optimizarea)
        # -------------------------------------------------------------------------
        if st.session_state.optimization_attempts > 0:
             st.info(f"Ultima rulare în Mod Optimă a necesitat **{st.session_state.optimization_attempts}** încercări.")


# TAB 3: Rezultate
with tab3:
    st.markdown("## 📊 Rezultate Generate")
    
    if not st.session_state.generated_variants:
        st.info("ℹ️ Nu există rezultate generate încă. Mergi la tab-ul 'Generează Random & Calculează Win' pentru a genera variante.")
    else:
        # Statistici
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Variante Generate", len(st.session_state.generated_variants))
        
        with col2:
            st.metric("Din Total", len(st.session_state.variants))
        
        with col3:
            st.metric("Runde Folosite", len(st.session_state.rounds_raw))

        with col4:
            st.metric("Scor WIN Obținut", st.session_state.win_score)
        
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
