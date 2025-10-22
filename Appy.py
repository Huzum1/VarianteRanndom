import streamlit as st
import pandas as pd
import random
from io import StringIO
import time

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

# Inițializare session state
if 'variants' not in st.session_state:
    st.session_state.variants = []
if 'generated_variants' not in st.session_state:
    st.session_state.generated_variants = []

def parse_variants(text):
    """Parse variantele din text"""
    variants = []
    errors = []
    
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
        
        variants.append({
            'id': variant_id,
            'combination': numbers
        })
    
    return variants, errors

def generate_sample_data(count=100):
    """Generează date de exemplu"""
    sample_data = []
    for i in range(1, count + 1):
        numbers = [str(random.randint(1, 49)) for _ in range(6)]
        sample_data.append(f"{i}, {' '.join(numbers)}")
    return '\n'.join(sample_data)

def variants_to_text(variants):
    """Convertește variantele în text"""
    return '\n'.join([f"{v['id']}, {v['combination']}" for v in variants])

def variants_to_csv(variants):
    """Convertește variantele în CSV"""
    df = pd.DataFrame(variants)
    return df.to_csv(index=False)

# Header
st.markdown("# 🎲 Generator Variante Loterie")
st.markdown("### Gestionează și generează variante aleatorii pentru loterie")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Statistici")
    st.metric("Variante Încărcate", len(st.session_state.variants))
    st.metric("Variante Generate", len(st.session_state.generated_variants))
    
    st.markdown("---")
    st.markdown("## ℹ️ Informații")
    st.info("""
    **Format acceptat:**
    ```
    ID, numere separate prin spațiu
    ```
    
    **Exemplu:**
    ```
    1, 5 7 44 32 18
    2, 12 23 34 45 49
    ```
    """)
    
    st.markdown("---")
    if st.button("🗑️ Resetează Tot", use_container_width=True):
        st.session_state.variants = []
        st.session_state.generated_variants = []
        st.rerun()

# Tabs principale
tab1, tab2, tab3 = st.tabs(["📝 Încarcă Variante", "🎲 Generează Random", "📊 Rezultate"])

# TAB 1: Încărcare Variante
with tab1:
    st.markdown("## 📝 Pas 1: Încarcă Variantele Tale")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Introdu variantele")
        st.caption("Format: ID, numere separate prin spațiu (ex: 1, 5 7 44 32 18)")
    
    with col2:
        if st.button("✨ Generează Date Exemplu", use_container_width=True):
            sample = generate_sample_data(100)
            st.session_state.sample_data = sample
            st.success("✅ S-au generat 100 variante exemplu!")
    
    # Textarea pentru input
    default_value = st.session_state.get('sample_data', '')
    variants_input = st.text_area(
        "Variante",
        value=default_value,
        height=300,
        placeholder="Exemplu:\n1, 5 7 44 32 18\n2, 12 23 34 45 49\n3, 1 2 3 4 5",
        label_visibility="collapsed"
    )
    
    # Butoane de acțiune
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("📥 Încarcă Variante", use_container_width=True, type="primary"):
            if not variants_input.strip():
                st.error("❌ Te rog să introduci variante!")
            else:
                with st.spinner("Se încarcă variantele..."):
                    variants, errors = parse_variants(variants_input)
                    
                    if variants:
                        st.session_state.variants = variants
                        st.success(f"✅ S-au încărcat {len(variants)} variante cu succes!")
                        
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
        # Upload fișier
        uploaded_file = st.file_uploader(
            "Sau încarcă fișier TXT/CSV",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            variants, errors = parse_variants(content)
            
            if variants:
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
        st.markdown("### 👀 Previzualizare Variante Încărcate")
        
        # Afișare primele și ultimele 5 variante
        df_preview = pd.DataFrame(st.session_state.variants)
        
        if len(st.session_state.variants) > 10:
            st.markdown(f"**Primele 5 variante din {len(st.session_state.variants)}:**")
            st.dataframe(df_preview.head(5), use_container_width=True, hide_index=True)
            
            st.markdown(f"**Ultimele 5 variante:**")
            st.dataframe(df_preview.tail(5), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_preview, use_container_width=True, hide_index=True)

# TAB 2: Generare Random
with tab2:
    st.markdown("## 🎲 Pas 2: Generează Variante Random")
    
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
                value=min(100, len(st.session_state.variants)),
                step=1
            )
            
            st.caption(f"Poți genera între 1 și {len(st.session_state.variants)} variante")
        
        with col2:
            st.markdown("### ")
            st.markdown("### ")
            if st.button("🎲 Generează Random", use_container_width=True, type="primary"):
                with st.spinner(f"Se generează {count} variante random..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Generare random fără duplicate
                    indices = list(range(len(st.session_state.variants)))
                    random.shuffle(indices)
                    selected_indices = indices[:count]
                    
                    # Simulare progres
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
        st.info("ℹ️ Nu există rezultate generate încă. Mergi la tab-ul 'Generează Random' pentru a genera variante.")
    else:
        # Statistici
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Variante Generate",
                len(st.session_state.generated_variants),
                delta=None
            )
        
        with col2:
            st.metric(
                "Din Total",
                len(st.session_state.variants),
                delta=None
            )
        
        with col3:
            percentage = (len(st.session_state.generated_variants) / len(st.session_state.variants)) * 100
            st.metric(
                "Procent",
                f"{percentage:.1f}%",
                delta=None
            )
        
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
            if st.button("🔄 Generează Din Nou", use_container_width=True):
                st.session_state.generated_variants = []
                st.rerun()
        
        st.markdown("---")
        
        # Afișare rezultate
        st.markdown("### 📋 Lista Completă de Variante Generate")
        
        df_results = pd.DataFrame(st.session_state.generated_variants)
        
        # Editor de date interactiv
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Opțiune de filtrare
        with st.expander("🔍 Caută în rezultate"):
            search_term = st.text_input("Caută după ID sau combinație")
            
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
