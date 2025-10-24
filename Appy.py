import streamlit as st
import pandas as pd
import random
from io import StringIO
import time
from collections import Counter

# Configurare pagină
st.set_page_config(
    page_title="Generator Variante Loterie cu Pattern-uri",
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
    .pattern-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem;
        background: #667eea;
        color: white;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCȚII PENTRU PATTERN-URI KENO RAPIDO 12/66
# ============================================================================

def is_prime(n):
    """Verifică dacă un număr este prim"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def calculate_pattern_score(numere):
    """
    Calculează scorul unei variante bazat pe toate cele 10 pattern-uri
    Scor mai mare = mai bună conformitatea cu pattern-urile
    """
    score = 0
    sorted_nums = sorted(numere)
    
    # PATTERN 1: DISTRIBUȚIE PE ZONE (6 zone x 11 numere)
    zone_counts = [
        sum(1 for n in numere if 1 <= n <= 11),
        sum(1 for n in numere if 12 <= n <= 22),
        sum(1 for n in numere if 23 <= n <= 33),
        sum(1 for n in numere if 34 <= n <= 44),
        sum(1 for n in numere if 45 <= n <= 55),
        sum(1 for n in numere if 56 <= n <= 66)
    ]
    
    for count in zone_counts:
        if 1 <= count <= 3:
            score += 10
        elif count == 0 or count == 4:
            score += 3
    
    if all(c > 0 for c in zone_counts):
        score += 15
    
    # PATTERN 2: PARITATE (PAR/IMPAR)
    pare = sum(1 for n in numere if n % 2 == 0)
    impare = 12 - pare
    
    if 5 <= pare <= 7 and 5 <= impare <= 7:
        score += 30
        if pare == 6 and impare == 6:
            score += 10
    elif 4 <= pare <= 8 and 4 <= impare <= 8:
        score += 15
    
    # PATTERN 3: NUMERE CONSECUTIVE
    consecutive_pairs = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i+1] - sorted_nums[i] == 1:
            consecutive_pairs += 1
    
    if 1 <= consecutive_pairs <= 3:
        score += 25
        if consecutive_pairs == 2:
            score += 10
    elif consecutive_pairs == 0 or consecutive_pairs == 4:
        score += 10
    
    # PATTERN 4: SUMA TOTALĂ
    suma = sum(numere)
    
    if 380 <= suma <= 420:
        score += 40
    elif 351 <= suma <= 470:
        score += 25
    elif 320 <= suma <= 500:
        score += 10
    
    # PATTERN 5: DECADE
    decade_counts = [
        sum(1 for n in numere if 1 <= n <= 10),
        sum(1 for n in numere if 11 <= n <= 20),
        sum(1 for n in numere if 21 <= n <= 30),
        sum(1 for n in numere if 31 <= n <= 40),
        sum(1 for n in numere if 41 <= n <= 50),
        sum(1 for n in numere if 51 <= n <= 60),
        sum(1 for n in numere if 61 <= n <= 66)
    ]
    
    for count in decade_counts[:6]:
        if 1 <= count <= 3:
            score += 5
    
    if 0 <= decade_counts[6] <= 2:
        score += 5
    
    # PATTERN 6: NUMERE PRIME
    prime_count = sum(1 for n in numere if is_prime(n))
    
    if 3 <= prime_count <= 4:
        score += 25
    elif 2 <= prime_count <= 5:
        score += 15
    elif 1 <= prime_count <= 6:
        score += 5
    
    # PATTERN 7: DISPERSIE (GAP-uri)
    gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
    gap_mediu = sum(gaps) / len(gaps)
    gap_maxim = max(gaps)
    
    if 4.5 <= gap_mediu <= 6.5:
        score += 25
    elif 3.5 <= gap_mediu <= 7.5:
        score += 10
    
    if gap_maxim <= 20:
        score += 10
    elif gap_maxim <= 30:
        score += 5
    
    # PATTERN 8: EXTREMITĂȚI (MICI-MIJLOC-MARI)
    mici = sum(1 for n in numere if 1 <= n <= 22)
    mijloc = sum(1 for n in numere if 23 <= n <= 44)
    mari = sum(1 for n in numere if 45 <= n <= 66)
    
    if 3 <= mici <= 5 and 3 <= mijloc <= 5 and 3 <= mari <= 5:
        score += 30
        if mici == 4 and mijloc == 4 and mari == 4:
            score += 15
    elif 2 <= mici <= 6 and 2 <= mijloc <= 6 and 2 <= mari <= 6:
        score += 15
    
    # PATTERN 9: CIFRA FINALĂ
    terminatii = [n % 10 for n in numere]
    max_same_ending = max(Counter(terminatii).values())
    
    if max_same_ending == 2 or max_same_ending == 3:
        score += 20
    elif max_same_ending == 4:
        score += 10
    
    preferred_endings = sum(1 for n in numere if n % 10 in [1, 3, 6])
    if preferred_endings >= 4:
        score += 10
    
    return score

def get_pattern_details(numere):
    """Returnează detalii despre pattern-urile unei variante"""
    sorted_nums = sorted(numere)
    
    # Zone
    zone_counts = [
        sum(1 for n in numere if 1 <= n <= 11),
        sum(1 for n in numere if 12 <= n <= 22),
        sum(1 for n in numere if 23 <= n <= 33),
        sum(1 for n in numere if 34 <= n <= 44),
        sum(1 for n in numere if 45 <= n <= 55),
        sum(1 for n in numere if 56 <= n <= 66)
    ]
    
    # Paritate
    pare = sum(1 for n in numere if n % 2 == 0)
    impare = 12 - pare
    
    # Consecutive
    consecutive = sum(1 for i in range(len(sorted_nums)-1) 
                     if sorted_nums[i+1] - sorted_nums[i] == 1)
    
    # Suma
    suma = sum(numere)
    
    # Prime
    prime_count = sum(1 for n in numere if is_prime(n))
    
    # Dispersie
    gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
    gap_mediu = sum(gaps) / len(gaps)
    
    # Extremități
    mici = sum(1 for n in numere if 1 <= n <= 22)
    mijloc = sum(1 for n in numere if 23 <= n <= 44)
    mari = sum(1 for n in numere if 45 <= n <= 66)
    
    return {
        'zone': zone_counts,
        'paritate': f"{pare}P-{impare}I",
        'consecutive': consecutive,
        'suma': suma,
        'prime': prime_count,
        'gap_mediu': round(gap_mediu, 2),
        'distributie': f"{mici}-{mijloc}-{mari}"
    }

def get_score_rating(score):
    """Returnează rating-ul bazat pe scor"""
    if score >= 320:
        return "🏆 PERFECT", "#FFD700"
    elif score >= 300:
        return "⭐ EXCELENT", "#32CD32"
    elif score >= 280:
        return "✅ FOARTE BUN", "#00CED1"
    elif score >= 250:
        return "👍 BUN", "#4169E1"
    elif score >= 200:
        return "⚠️ ACCEPTABIL", "#FFA500"
    else:
        return "❌ SLAB", "#FF6347"

# ============================================================================
# FUNCȚII ORIGINALE
# ============================================================================

# Inițializare session state
if 'variants' not in st.session_state:
    st.session_state.variants = []
if 'generated_variants' not in st.session_state:
    st.session_state.generated_variants = []
if 'pattern_filter_enabled' not in st.session_state:
    st.session_state.pattern_filter_enabled = True

def parse_variants(text):
    """Parse variantele din text"""
    variants = []
    errors = []
    
    lines = text.strip().split('\n')
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
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
        
        # Parsează numerele pentru verificare
        try:
            num_list = [int(x.strip()) for x in numbers.replace(',', ' ').split()]
            variants.append({
                'id': variant_id,
                'combination': numbers,
                'numbers': num_list
            })
        except:
            errors.append(f"Linia {i}: Numere invalide")
            continue
    
    return variants, errors

def generate_sample_data(count=100):
    """Generează date de exemplu pentru Keno Rapido 12/66"""
    sample_data = []
    for i in range(1, count + 1):
        numbers = sorted(random.sample(range(1, 67), 12))
        sample_data.append(f"{i}, {' '.join(map(str, numbers))}")
    return '\n'.join(sample_data)

def variants_to_text(variants):
    """Convertește variantele în text"""
    return '\n'.join([f"{v['id']}, {v['combination']}" for v in variants])

def variants_to_csv(variants):
    """Convertește variantele în CSV cu detalii pattern-uri"""
    data = []
    for v in variants:
        row = {
            'ID': v['id'],
            'Combinație': v['combination']
        }
        if 'score' in v:
            row['Scor'] = v['score']
            details = get_pattern_details(v['numbers'])
            row['Zone'] = '-'.join(map(str, details['zone']))
            row['Paritate'] = details['paritate']
            row['Consecutive'] = details['consecutive']
            row['Suma'] = details['suma']
            row['Prime'] = details['prime']
            row['Gap Mediu'] = details['gap_mediu']
            row['Distribuție'] = details['distributie']
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# ============================================================================
# INTERFAȚĂ STREAMLIT
# ============================================================================

# Header
st.markdown("# 🎲 Generator Variante Loterie cu Pattern-uri")
st.markdown("### Keno Rapido 12/66 - Selectare inteligentă bazată pe 10 pattern-uri")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Statistici")
    st.metric("Variante Încărcate", len(st.session_state.variants))
    st.metric("Variante Generate", len(st.session_state.generated_variants))
    
    st.markdown("---")
    st.markdown("## ⚙️ Setări Pattern-uri")
    
    st.session_state.pattern_filter_enabled = st.toggle(
        "Activează Filtru Pattern-uri",
        value=st.session_state.pattern_filter_enabled,
        help="Când este activ, variantele sunt sortate după scorul pattern-urilor"
    )
    
    if st.session_state.pattern_filter_enabled:
        st.info("✅ Variantele vor fi selectate după conformitatea cu pattern-urile!")
    else:
        st.warning("⚠️ Selecție complet aleatorie (fără filtrare)")
    
    st.markdown("---")
    st.markdown("## ℹ️ Pattern-uri")
    with st.expander("📋 Vezi Pattern-urile"):
        st.markdown("""
        **10 Pattern-uri Aplicate:**
        1. 🎯 Distribuție pe zone (6 zone)
        2. ⚡ Paritate (PAR/IMPAR)
        3. 🔗 Numere consecutive
        4. ➕ Suma totală (ideal: 380-420)
        5. 🔢 Decade echilibrate
        6. 🔐 Numere prime (3-4 ideal)
        7. 📏 Dispersie (gap mediu)
        8. 🎯 Extremități (4-4-4 ideal)
        9. #️⃣ Cifre finale
        10. 🔄 Repetiție (neaplicat)
        
        **Scor maxim: ~360 puncte**
        """)
    
    st.markdown("---")
    st.markdown("## ℹ️ Format")
    st.info("""
    **Format acceptat:**
    ```
    ID, numere separate prin spațiu
    ```
    
    **Exemplu Keno 12/66:**
    ```
    1, 5 7 12 23 34 44 45 50 56 60 63 66
    2, 1 10 17 24 25 33 37 42 48 53 60 63
    ```
    """)
    
    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.variants = []
        st.session_state.generated_variants = []
        st.rerun()

# Tabs principale
tab1, tab2, tab3, tab4 = st.tabs(["📝 Încarcă Variante", "🎲 Generează cu Pattern-uri", "📊 Rezultate", "📈 Analiză Pattern-uri"])

# TAB 1: Încărcare Variante
with tab1:
    st.markdown("## 📝 Pas 1: Încarcă Variantele Tale")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Introdu variantele (Keno Rapido 12/66)")
        st.caption("Format: ID, 12 numere între 1-66 separate prin spațiu")
    
    with col2:
        if st.button("✨ Generează 3000 Exemple", use_container_width=True):
            sample = generate_sample_data(3000)
            st.session_state.sample_data = sample
            st.success("✅ S-au generat 3000 variante exemplu!")
    
    # Textarea pentru input
    default_value = st.session_state.get('sample_data', '')
    variants_input = st.text_area(
        "Variante",
        value=default_value,
        height=300,
        placeholder="Exemplu:\n1, 5 7 12 23 34 44 45 50 56 60 63 66\n2, 1 10 17 24 25 33 37 42 48 53 60 63",
        label_visibility="collapsed"
    )
    
    # Butoane de acțiune
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("📥 Încarcă Variante", use_container_width=True, type="primary"):
            if not variants_input.strip():
                st.error("❌ Te rog să introduci variante!")
            else:
                with st.spinner("Se încarcă și se analizează variantele..."):
                    variants, errors = parse_variants(variants_input)
                    
                    if variants:
                        # Calculează scoruri pentru toate variantele
                        progress_bar = st.progress(0)
                        for i, v in enumerate(variants):
                            v['score'] = calculate_pattern_score(v['numbers'])
                            progress_bar.progress((i + 1) / len(variants))
                        
                        st.session_state.variants = variants
                        
                        # Statistici scoruri
                        scores = [v['score'] for v in variants]
                        avg_score = sum(scores) / len(scores)
                        
                        st.success(f"✅ S-au încărcat {len(variants)} variante cu succes!")
                        st.info(f"📊 Scor mediu: {avg_score:.2f} | Min: {min(scores)} | Max: {max(scores)}")
                        
                        if errors:
                            with st.expander("⚠️ Avertismente"):
                                for error in errors:
                                    st.warning(error)
                    else:
                        st.error("❌ Nu s-au putut încărca variante valide!")
                        if errors:
                            for error in errors:
                                st.error(error)
    
    with col2:
        uploaded_file = st.file_uploader(
            "Sau încarcă fișier TXT/CSV",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            with st.spinner("Se procesează fișierul..."):
                variants, errors = parse_variants(content)
                
                if variants:
                    # Calculează scoruri
                    for v in variants:
                        v['score'] = calculate_pattern_score(v['numbers'])
                    
                    st.session_state.variants = variants
                    st.success(f"✅ S-au încărcat {len(variants)} variante din fișier!")
                else:
                    st.error("❌ Fișierul nu conține variante valide!")
    
    with col3:
        if st.session_state.variants:
            st.download_button(
                "💾 Descarcă Variante",
                data=variants_to_text(st.session_state.variants),
                file_name="variante_complete.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Previzualizare variante încărcate
    if st.session_state.variants:
        st.markdown("---")
        st.markdown("### 👀 Previzualizare Variante Încărcate (cu Scoruri)")
        
        # Creează DataFrame cu scoruri
        preview_data = []
        for v in st.session_state.variants[:10]:
            rating, color = get_score_rating(v['score'])
            preview_data.append({
                'ID': v['id'],
                'Combinație': v['combination'],
                'Scor': v['score'],
                'Rating': rating
            })
        
        df_preview = pd.DataFrame(preview_data)
        
        st.dataframe(
            df_preview,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Scor": st.column_config.NumberColumn(
                    "Scor Pattern",
                    help="Scor bazat pe conformitatea cu pattern-urile",
                    format="%d ⭐"
                )
            }
        )
        
        if len(st.session_state.variants) > 10:
            st.caption(f"Se afișează primele 10 din {len(st.session_state.variants)} variante")

# TAB 2: Generare cu Pattern-uri
with tab2:
    st.markdown("## 🎲 Pas 2: Generează Variante cu Pattern-uri")
    
    if not st.session_state.variants:
        st.warning("⚠️ Nu există variante încărcate! Mergi la tab-ul 'Încarcă Variante' pentru a adăuga variante.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Ai {len(st.session_state.variants)} variante disponibile")
            
            count = st.number_input(
                "Câte variante să generez?",
                min_value=1,
                max_value=len(st.session_state.variants),
                value=min(1000, len(st.session_state.variants)),
                step=1
            )
            
            if st.session_state.pattern_filter_enabled:
                st.success(f"✅ Vor fi selectate cele mai bune {count} variante după scorul pattern-urilor!")
            else:
                st.info(f"ℹ️ Vor fi selectate {count} variante complet aleatoriu")
        
        with col2:
            st.markdown("### ")
            st.markdown("### ")
            if st.button("🎲 Generează Variante", use_container_width=True, type="primary"):
                with st.spinner(f"Se generează {count} variante..."):
                    progress_bar = st.progress(0)
                    
                    if st.session_state.pattern_filter_enabled:
                        # Sortează după scor și ia primele N
                        sorted_variants = sorted(
                            st.session_state.variants,
                            key=lambda x: x['score'],
                            reverse=True
                        )
                        
                        # Simulare progres
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        st.session_state.generated_variants = sorted_variants[:count]
                        
                        # Statistici
                        scores = [v['score'] for v in st.session_state.generated_variants]
                        avg_score = sum(scores) / len(scores)
                        
                        st.success(f"✅ S-au generat {len(st.session_state.generated_variants)} variante!")
                        st.info(f"📊 Scor mediu selecție: {avg_score:.2f} (Min: {min(scores)}, Max: {max(scores)})")
                        
                    else:
                        # Selecție aleatorie (fără duplicate)
                        indices = list(range(len(st.session_state.variants)))
                        random.shuffle(indices)
                        selected_indices = indices[:count]
                        
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        st.session_state.generated_variants = [
                            st.session_state.variants[i] for i in selected_indices
                        ]
                        
                        st.success(f"✅ S-au generat {len(st.session_state.generated_variants)} variante random!")
                    
                    st.balloons()

# TAB 3: Rezultate
with tab3:
    st.markdown("## 📊 Rezultate Generate")
    
    if not st.session_state.generated_variants:
        st.info("ℹ️ Nu există rezultate generate încă. Mergi la tab-ul 'Generează cu Pattern-uri'.")
    else:
        # Statistici
        col1, col2, col3, col4 = st.columns(4)
        
        scores = [v['score'] for v in st.session_state.generated_variants]
        avg_score = sum(scores) / len(scores)
        
        with col1:
            st.metric("Variante Generate", len(st.session_state.generated_variants))
        
        with col2:
            st.metric("Scor Mediu", f"{avg_score:.1f}")
        
        with col3:
            st.metric("Scor Maxim", max(scores))
        
        with col4:
            excellent = sum(1 for s in scores if s >= 280)
            st.metric("Variante Foarte Bune", f"{excellent} ({excellent/len(scores)*100:.1f}%)")
        
        st.markdown("---")
        
        # Butoane de export
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "💾 Descarcă TXT",
                data=variants_to_text(st.session_state.generated_variants),
                file_name=f"variante_pattern_{len(st.session_state.generated_variants)}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "📊 Descarcă CSV Complet",
                data=variants_to_csv(st.session_state.generated_variants),
                file_name=f"variante_pattern_detalii_{len(st.session_state.generated_variants)}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Generează Din Nou", use_container_width=True):
                st.session_state.generated_variants = []
                st.rerun()
        
        st.markdown("---")
        
        # Afișare rezultate cu detalii
        st.markdown("### 📋 Lista Completă de Variante Generate")
        
        display_data = []
        for v in st.session_state.generated_variants:
            rating, color = get_score_rating(v['score'])
            details = get_pattern_details(v['numbers'])
            
            display_data.append({
                'ID': v['id'],
                'Combinație': v['combination'],
                'Scor': v['score'],
                'Rating': rating,
                'Paritate': details['paritate'],
                'Suma': details['suma'],
                'Prime': details['prime'],
                'Dist.': details['distributie']
            })
        
        df_results = pd.DataFrame(display_data)
        
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Scor": st.column_config.NumberColumn(
                    "Scor",
                    help="Scor pattern (max 360)",
                    format="%d"
                ),
                "Suma": st.column_config.NumberColumn(
                    "Suma",
                    help="Suma numerelor (ideal: 380-420)"
                )
            }
        )
        
        # Opțiune de filtrare
        with st.expander("🔍 Caută și Filtrează"):
            col1, col2 = st.columns(2)
            
            with col1:
                search_term = st.text_input("Caută după ID sau numere")
            
            with col2:
                min_score = st.slider(
                    "Scor minim",
                    min_value=int(min(scores)),
                    max_value=int(max(scores)),
                    value=int(min(scores))
                )
            
            filtered = [
                v for v in st.session_state.generated_variants
                if (not search_term or 
                    search_term.lower() in v['id'].lower() or 
                    search_term in v['combination']) and
                   v['score'] >= min_score
            ]
            
            if filtered and (search_term or min_score > min(scores)):
                st.success(f"✅ S-au găsit {len(filtered)} rezultate")
                
                filtered_display = []
                for v in filtered:
                    rating, color = get_score_rating(v['score'])
                    filtered_display.append({
                        'ID': v['id'],
                        'Combinație': v['combination'],
                        'Scor': v['score'],
                        'Rating': rating
                    })
                
                st.dataframe(
                    pd.DataFrame(filtered_display),
                    use_container_width=True,
                    hide_index=True
                )

# TAB 4: Analiză Pattern-uri
with tab4:
    st.markdown("## 📈 Analiză Detaliată Pattern-uri")
    
    if not st.session_state.generated_variants:
        st.info("ℹ️ Generează mai întâi variante pentru a vedea analiza pattern-urilor.")
    else:
        st.markdown("### 📊 Distribuția Scorurilor")
        
        scores = [v['score'] for v in st.session_state.generated_variants]
        
        # Histogramă scoruri
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(scores, bins=20, color='#667eea', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=2, label=f'Media: {np.mean(scores):.1f}')
        ax.set_xlabel('Scor Pattern')
        ax.set_ylabel('Număr Variante')
        ax.set_title('Distribuția Scorurilor Pattern-uri')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Statistici detaliate per pattern
        st.markdown("### 🎯 Statistici per Pattern")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Paritate (PAR-IMPAR)")
            paritate_counts = {}
            for v in st.session_state.generated_variants:
                details = get_pattern_details(v['numbers'])
                key = details['paritate']
                paritate_counts[key] = paritate_counts.get(key, 0) + 1
            
            for key, count in sorted(paritate_counts.items(), key=lambda x: x[1], reverse=True):
                proc = (count / len(st.session_state.generated_variants)) * 100
                st.write(f"**{key}**: {count} variante ({proc:.1f}%)")
        
        with col2:
            st.markdown("#### Distribuție Extremități")
            dist_counts = {}
            for v in st.session_state.generated_variants:
                details = get_pattern_details(v['numbers'])
                key = details['distributie']
                dist_counts[key] = dist_counts.get(key, 0) + 1
            
            for key, count in sorted(dist_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                proc = (count / len(st.session_state.generated_variants)) * 100
                st.write(f"**{key}**: {count} variante ({proc:.1f}%)")
        
        st.markdown("---")
        
        # Top 10 variante
        st.markdown("### 🏆 Top 10 Cele Mai Bune Variante")
        
        top_variants = sorted(
            st.session_state.generated_variants,
            key=lambda x: x['score'],
            reverse=True
        )[:10]
        
        for i, v in enumerate(top_variants, 1):
            with st.expander(f"#{i} - {v['id']} (Scor: {v['score']})"):
                rating, color = get_score_rating(v['score'])
                st.markdown(f"**Rating**: {rating}")
                st.markdown(f"**Numere**: {v['combination']}")
                
                details = get_pattern_details(v['numbers'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Paritate", details['paritate'])
                    st.metric("Suma", details['suma'])
                
                with col2:
                    st.metric("Consecutive", details['consecutive'])
                    st.metric("Prime", details['prime'])
                
                with col3:
                    st.metric("Gap Mediu", details['gap_mediu'])
                    st.metric("Distribuție", details['distributie'])
                
                st.markdown(f"**Zone**: {'-'.join(map(str, details['zone']))}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>🎲 Generator Variante Loterie cu Pattern-uri | Keno Rapido 12/66 | 10 Pattern-uri Integrate</p>
    </div>
    """,
    unsafe_allow_html=True
)
