        # --- Vizualizare Frecvență (Păstrată) ---
        col_list, col_chart = st.columns([1, 2])
        
        with col_list:
            st.markdown("### 📋 Lista de Variante")
            df_results = pd.DataFrame(st.session_state.generated_variants)
            # Descarcă TXT (folosind variants_to_text) - ACESTA ESTE BUTONUL DORIT
            st.download_button("💾 Descarcă TXT", data=variants_to_text(st.session_state.generated_variants), file_name=f"variante_optim_{len(st.session_state.generated_variants)}.txt", mime="text/plain", use_container_width=True)
            # Descarcă CSV (folosind variants_to_csv)
            st.download_button("📊 Descarcă CSV", data=variants_to_csv(st.session_state.generated_variants), file_name=f"variante_optim_{len(st.session_state.generated_variants)}.csv", mime="text/csv", use_container_width=True)
            st.dataframe(df_results, use_container_width=True, hide_index=True, height=250)
