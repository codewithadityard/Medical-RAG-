import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Medical Diagnostic Engine", page_icon="🩺", layout="wide")
st.title("🩺 AI-Powered Rare Disease Diagnostic Engine")
st.markdown("Enter clinical symptoms in plain English to query the GNN and retrieve literature evidence.")

# Input Section
symptoms_input = st.text_input(
    "Patient Symptoms (comma-separated):", 
    placeholder="e.g., abnormal heart morphology, spider fingers",
    help="Enter human-readable symptoms. The engine will map these to HPO IDs."
)

if st.button("Run Diagnostic Analysis", type="primary"):
    if not symptoms_input:
        st.warning("Please enter at least one symptom.")
    else:
        # Clean and format the symptoms into a list
        symptoms_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]
        
        with st.spinner("Initializing GNN & Fetching PubMed Evidence..."):
            try:
                # Call the FastAPI backend
                response = requests.post(
                    "http://localhost:8000/diagnose", 
                    json={"symptoms": symptoms_list}
                )
                data = response.json()

                if isinstance(data, str):
                    st.error(data)
                
                elif "error" in data:
                    st.error(f"Error: {data['error']}")
                else:
                    # Layout Columns for Results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("🧬 GNN Top Predictions")
                        for idx, disease in enumerate(data["top_diseases"]):
                            st.write(f"**{idx + 1}.** {disease}")
                            
                        st.subheader("🛡️ MedNLI Verification")
                        if data["is_faithful"]:
                            st.success("✅ **Verified:** LLM summary is mathematically entailed by PubMed literature.")
                        else:
                            st.error("🚨 **Blocked:** LLM hallucination detected. Output contradicts literature.")

                    with col2:
                        st.subheader("📝 Clinical Synthesis")
                        st.info(data["summary"])
                        
                        with st.expander("📚 View Retrieved PubMed Evidence"):
                            for i, doc in enumerate(data["evidence"]):
                                st.markdown(f"**Document {i+1}**\n\n{doc}\n")
                                st.divider()
                                
            except requests.exceptions.ConnectionError:
                st.error("Backend connection failed. Ensure FastAPI is running on port 8000.")