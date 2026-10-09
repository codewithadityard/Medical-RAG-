import streamlit as st
import requests
import base64

# Configure page
st.set_page_config(page_title="Multimodal Rare Disease Diagnostic Engine", page_icon="🩺", layout="wide")
st.title("🩺 Multimodal Rare Disease Diagnostic Engine")
st.markdown("Enter clinical symptoms in plain English and/or upload clinical imagery (X-rays, dermatology, eye scans) to query the GNN and retrieve literature evidence.")

# Custom CSS to style the upload button red
st.markdown(
    """
    <style>
    div[data-testid="stFileUploader"] button,
    .stFileUploader button {
        background-color: #dc3545 !important;
        color: white !important;
        border: 1px solid #dc3545 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stFileUploader"] button:hover,
    .stFileUploader button:hover {
        background-color: #bd2130 !important;
        border-color: #b21f2d !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for image uploads
with st.sidebar:
    st.header("Clinical Imagery")
    uploaded_image = st.file_uploader(
        "Clinical Image (Optional):",
        type=["jpg", "jpeg", "png"],
        help="Microsoft BiomedCLIP will extract visual clinical phenotypes from this image."
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# Main Input Section
symptoms_input = st.text_input(
    "Patient Symptoms (comma-separated):", 
    placeholder="e.g., unusually long fingers, sunken chest",
    help="Enter human-readable symptoms. The engine will map these to HPO IDs via SapBERT."
)

if st.button("Run Diagnostic Analysis", type="primary"):
    if not symptoms_input and uploaded_image is None:
        st.warning("Please enter at least one symptom or upload a clinical image.")
    else:
        # Clean and format the symptoms into a list
        symptoms_list = [s.strip() for s in symptoms_input.split(",") if s.strip()] if symptoms_input else []
        
        # Encode image to base64 if present
        image_b64 = None
        if uploaded_image is not None:
            image_bytes = uploaded_image.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        with st.spinner("Analyzing with BiomedCLIP, GNN & Fetching PubMed Evidence..."):
            try:
                # Call the FastAPI backend
                payload = {"symptoms": symptoms_list}
                if image_b64:
                    payload["image_base64"] = image_b64

                response = requests.post("http://localhost:8000/diagnose", json=payload)
                data = response.json()

                if isinstance(data, str):
                    st.error(data)
                elif "error" in data:
                    st.error(f"Error: {data['error']}")
                else:
                    # Layout Columns for Results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if data.get("visual_detections"):
                            st.subheader("🖼️ BiomedCLIP Visual Detections")
                            for d in data["visual_detections"]:
                                st.write(f"- **{d['phenotype']}** (Confidence: {d['confidence']*100:.1f}%)")
                        
                        st.subheader("🧬 GNN Top Predictions")
                        for idx, disease in enumerate(data["top_diseases"]):
                            st.write(f"**{idx + 1}.** {disease}")
                            
                        st.subheader("🛡️ Sentence MedNLI Verification")
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