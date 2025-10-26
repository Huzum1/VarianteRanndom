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
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #667eea;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: white !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    h2, h3 {
        color: #667eea !important;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
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
if 'rounds_raw' not in st.session_state: # NOU: Păstrează rundele ca set de numere pentru parsare ușoară
    st.session_state.rounds_raw = []
if 'win_score' not in st.session_state:
    st.session_state.win_score = 0
if 'round_performance_df' not in st.session_state: # NOU: DataFrame pentru performanța rundelor
    st.session_state.round_performance_df = pd.DataFrame()


def clean_variant_combination(numbers_str):
    """
    Curăță șirul de numere, asigură unicitatea (elimină duplicatele din aceeași variantă)
    și le sortează. Returnează combinația curățată și numărul de duplicate eliminate.
    """
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
    """
    Parse variantele din text, curățând duplicatele din interiorul fiecărei combinații.
    """
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

        variants.append({
            'id': variant_id,
            'combination': cleaned_combination
        })
    
    # Etapa finală: Elimină duplicatele între variante
    df = pd.DataFrame(variants)
    if not df.empty:
        df_unique = df.drop_duplicates(subset=['combination']).reset_index(drop=True)
        # Reatribuie ID-urile serial
        df_unique['id'] = (df_unique.index + 1).astype(str)
        
        final_variants = df_unique.to_dict('records')
        
        total_inter_duplicates_removed = len(variants) - len(final_variants)
    else:
        final_variants = []
        total_inter_duplicates_removed = 0
    
    return final_variants, errors, total_internal_duplicates_removed, total_inter_duplicates_removed

@st.cache_data
def parse_rounds(rounds_file):
    """Procesează fișierul de runde și returnează o listă de seturi (runda) și o listă de string-uri (afisare)."""
    if rounds_file is None:
        return [], []
    
    rounds_set_list = []
    rounds_display_list = []
    
    try:
        content = rounds_file.getvalue().decode("utf-8")
        
        for line in content.splitlines():
            parts = [p.strip() for p in line.replace(',', ' ').split() if p.strip().isdigit()]
            round_numbers = {int(p) for p in parts if p.isdigit()} 
            
            if len(round_numbers) >= 4:
                rounds_set_list.append(round_numbers)
                # Formatează numerele pentru afișare (sortate)
                display_numbers = ' '.join(map(str, sorted(list(round_numbers))))
                rounds_display_list.append(display_numbers)
                
        return rounds_set_list, rounds_display_list
    except Exception as e:
        st.error(f"Eroare la procesarea rundelor: {e}")
        return [], []

def calculate_wins(generated_variants, rounds):
    """Calculează numărul total de potriviri (4/4, 5/5, etc.)."""
    if not rounds or not generated_variants:
        return 0
    
    total_wins = 0
    
    for variant_data in generated_variants:
        try:
            variant_numbers_list = [int(n) for n in variant_data['combination'].split() if n.isdigit()]
            variant_set = set(variant_numbers_list)
        except:
            continue
        
        for runda in rounds:
            if variant_set.issubset(runda):
                total_wins += 1
                
    return total_wins

def analyze_round_performance(generated_variants, rounds_set, rounds_display):
    """
    NOU: Calculează performanța fiecărei runde individuale pe baza variantelor generate.
    Returnează un DataFrame cu Runda și Scor.
    """
    if not rounds_set or not generated_variants:
        return pd.DataFrame({'Runda': rounds_display, 'WINs': [0] * len(rounds_display)})

    variant_sets = []
    for variant_data in generated_variants:
        try:
            variant_sets.append(set(int(n) for n in variant_data['combination'].split() if n.isdigit()))
        except:
            continue

    if not variant_sets:
         return pd.DataFrame({'Runda': rounds_display, 'WINs': [0] * len(rounds_display)})

    performance_data = []
    for i, runda_set in enumerate(rounds_set):
        wins_in_round = 0
        for v_set in variant_sets:
            # Dacă toate numerele din varianta generată se regăsesc în rundă
            if v_set.issubset(runda_set):
                wins_in_round += 1
        
        round_name = f"Runda {i+1}"
        round_display_str = f"({rounds_display[i]})"
        
        # Formatul cerut: Runda 1 - 3 variante câștigătoare
        win_label = f" - {wins_in_round} variante câștigătoare"
        
        performance_data.append({
            'ID': i + 1,
            'Runda': round_name,
            'Numere': round_display_str,
            'WINs': wins_in_round,
            'Rezultat': round_name + win_label
        })
        
    return pd.DataFrame(performance_data)


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
        st.session_state.round_performance_df = pd.DataFrame()
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante & Curăță", "🎲 Generează Random & Calculează Win", "📊 Rezultate"])

# TAB 1: Încărcare Variante (Păstrat neschimbat)
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

# TAB 2: Generare Random & Calcul WIN (Modificat)
with tab2:
    st.markdown("## 🎲 Pas 2: Generează Variante Random & Calculează Performanța")
    
    if not st.session_state.variants:
        st.warning("⚠️ Nu există variante curățate încă! Mergi la tab-ul 'Încarcă Variante & Curăță'.")
        
    else:
        # -------------------------------------------------------------------------
        # Secțiunea 1: Încărcare Runde
        # -------------------------------------------------------------------------
        st.markdown("### 1. Încarcă Rundele (Extragerile) de Bază")
        
        col_rounds, col_rounds_info = st.columns([2, 1])
        
        with col_rounds:
            rounds_file = st.file_uploader(
                "Încărcați fișierul cu Rundele (extragerile)",
                type=['txt', 'csv'],
                key="rounds_uploader"
            )

        if rounds_file:
            rounds_set_list, rounds_display_list = parse_rounds(rounds_file)
            st.session_state.rounds = rounds_set_list
            st.session_state.rounds_raw = rounds_display_list
            
            with col_rounds_info:
                st.metric("Total Runde Încărcate", len(st.session_state.rounds))
        
        
        # -------------------------------------------------------------------------
        # Secțiunea 2: Previzualizare Runde ȘI Performanță
        # -------------------------------------------------------------------------
        if st.session_state.rounds_raw and not st.session_state.round_performance_df.empty:
            
            st.markdown("#### 🎯 Previzualizare Performanță Runde (WINs pe runda)")
            
            df_perf = st.session_state.round_performance_df
            
            # Afișează doar coloanele relevante
            df_display = df_perf[['Runda', 'Numere', 'Rezultat']].rename(columns={'Rezultat': 'Analiză WIN'})
            
            # 1. Afișează primele 10 rânduri (Preview)
            st.dataframe(
                df_display.head(10),
                use_container_width=True,
                hide_index=True,
                height=370
            )

            # 2. Buton pentru afișarea completă (cu scroll)
            if len(df_perf) > 10:
                if st.button(f"Vizualizează Toate Rundele ({len(df_perf)})", use_container_width=True):
                    with st.expander("Lista completă de Runde și Scoruri", expanded=True):
                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                            height=600 # Limită înălțimea pentru a permite scroll
                        )

        st.markdown("---")
        
        # -------------------------------------------------------------------------
        # Secțiunea 3: Generare Random & Calcul
        # -------------------------------------------------------------------------
        st.markdown("### 3. Generare Eșantion Aleatoriu & Calcul Score")

        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"Ai **{len(st.session_state.variants)}** variante unice disponibile.")
            
            count = st.number_input(
                "Câte variante să generez?",
                min_value=1,
                max_value=len(st.session_state.variants),
                value=min(1165, len(st.session_state.variants)),
                step=1
            )
        
        with col2:
            st.markdown("### ")
            st.markdown("### ")
            if st.button("🎲 Generează Random & Calculează", use_container_width=True, type="primary"):
                
                if not st.session_state.rounds:
                    st.warning("Vă rugăm să încărcați rundele mai întâi.")
                
                else:
                    with st.spinner(f"Se generează {count} variante random și se calculează scorul..."):
                        
                        progress_bar = st.progress(0)
                        
                        # Generare random
                        indices = list(range(len(st.session_state.variants)))
                        random.shuffle(indices)
                        selected_indices = indices[:count]
                        
                        generated_variants = [
                            st.session_state.variants[i] for i in selected_indices
                        ]
                        st.session_state.generated_variants = generated_variants
                        
                        # Calcul Score WIN Total
                        win_score = calculate_wins(generated_variants, st.session_state.rounds)
                        st.session_state.win_score = win_score
                        
                        # Calcul Performanță pe Rundă (DataFrame)
                        df_perf = analyze_round_performance(
                            generated_variants, 
                            st.session_state.rounds, 
                            st.session_state.rounds_raw
                        )
                        st.session_state.round_performance_df = df_perf
                        
                        win_message = f"✅ S-au generat {len(generated_variants)} variante și s-au obținut **{win_score} WINs**!"
                        
                        # Simulare progres
                        for i in range(100):
                            time.sleep(0.005)
                            progress_bar.progress(i + 1)
                        
                        st.success(win_message)
                        st.balloons()
                        st.rerun() # Re-rulează pentru a afișa imediat previzualizarea rundelor
        
        with col3:
            st.markdown("### ")
            st.markdown("### ")
            st.metric(
                "Scor de Performanță Total", 
                st.session_state.win_score
            )


# TAB 3: Rezultate (Păstrat neschimbat)
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
