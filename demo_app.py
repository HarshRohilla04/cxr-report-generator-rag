"""
Demo UI — Chest X-ray Report Generation
Gradio-based web interface for the full pipeline.
Usage: .venv/Scripts/python.exe demo_app.py
"""

import gradio as gr
import sys
import os
import base64
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline_standalone import MedicalImagingPipeline

# ============================================================================
# GLOBAL PIPELINE (loaded once on startup)
# ============================================================================
pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is None:
        print("Checking for core data dependencies...")
        
        # Verify core data exists before trying to load
        missing = []
        if not Path("models/multilabel_classifier_biomedclip_v2.pt").exists():
            missing.append("models/multilabel_classifier_biomedclip_v2.pt")
        if not Path("embeddings").exists() or not any(Path("embeddings").iterdir()):
            missing.append("embeddings/ (directory is missing or empty)")
        if not Path("faiss_index").exists() or not any(Path("faiss_index").iterdir()):
            missing.append("faiss_index/ (directory is missing or empty)")
            
        if missing:
            error_msg = (
                "\n" + "="*80 + "\n"
                "🚨 CRITICAL ERROR: MISSING CORE DATA 🚨\n\n"
                "The pipeline cannot start because the following required files/directories are missing:\n"
                + "\n".join(f"  - {m}" for m in missing) + "\n\n"
                "HOW TO FIX THIS:\n"
                "1. Go to the GitHub repository's 'Releases' page.\n"
                "2. Download the 'core-data.zip' file.\n"
                "3. Extract it directly into the 'major-project' folder.\n"
                "4. Restart this app.\n"
                + "="*80 + "\n"
            )
            print(error_msg)
            sys.exit(1)
            
        print("Loading pipeline (first run)...")
        pipeline = MedicalImagingPipeline(llm_provider="openai")
    return pipeline

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_xray(image_path):
    """Run full pipeline on uploaded image and format results for Gradio."""
    if image_path is None:
        return "⚠️ Please upload a chest X-ray image.", "", "", ""
    
    pipe = load_pipeline()
    
    try:
        result = pipe.predict(image_path, retrieve_k=5, ablation_mode='full')
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""
    
    # --- Format: Generated Report ---
    report = result.get('generated_report', 'No report generated.')
    
    # --- Format: Abnormality Assessment ---
    abn_score = result['abnormality_score']
    abn_label = result['abnormality_label'].upper()
    override = result.get('override_reason')
    
    if abn_label == 'NORMAL':
        abn_icon = "🟢"
        abn_color = "Normal"
    elif abn_label == 'BORDERLINE':
        abn_icon = "🟡"
        abn_color = "Borderline"
    else:
        abn_icon = "🔴"
        abn_color = "Abnormal"
    
    assessment = f"## {abn_icon} {abn_color}\n\n"
    assessment += f"**Abnormality Score:** `{abn_score:.4f}`\n\n"
    
    if override:
        assessment += f"⚠️ *Override: {override}*\n\n"
    
    assessment += "---\n\n"
    assessment += "### Disease Probabilities\n\n"
    assessment += "| Disease | Probability | Status |\n"
    assessment += "|---------|:-----------:|:------:|\n"
    
    # Sort diseases by probability
    sorted_diseases = sorted(
        result['disease_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for disease, prob in sorted_diseases:
        if prob >= 0.50:
            status = "🔴 Detected"
        elif prob >= 0.30:
            status = "🟡 Possible"
        else:
            status = "⚪ Unlikely"
        
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        assessment += f"| {disease} | `{bar}` {prob:.1%} | {status} |\n"
    
    # --- Format: Retrieved Evidence (HTML with images) ---
    retrieved = result.get('retrieved_cases', {})
    abn_cases = retrieved.get('abnormal', [])
    healthy_cases = retrieved.get('healthy', [])
    
    evidence_html = ""
    
    if abn_cases:
        evidence_html += '<h3 style="color:#ef4444;">🔬 Similar Abnormal Cases</h3>'
        for i, case in enumerate(abn_cases[:3], 1):
            evidence_html += _case_card_html(case, f"Case {i}", "#ef4444")
    
    if healthy_cases:
        evidence_html += '<h3 style="color:#22c55e; margin-top:1.5rem;">🟢 Healthy Reference Cases</h3>'
        for i, case in enumerate(healthy_cases[:2], 1):
            evidence_html += _case_card_html(case, f"Reference {i}", "#22c55e")
    
    if not evidence_html:
        evidence_html = '<p style="color:#888;">No retrieved evidence available.</p>'
    
    return report, assessment, evidence_html, f"✅ Analysis complete — {abn_label}"


def _case_card_html(case, title, accent_color):
    """Build an HTML card for a retrieved case, including its image if available."""
    card = f'<div style="border:1px solid {accent_color}33; border-radius:12px; padding:1rem; margin:0.75rem 0; display:flex; gap:1rem; align-items:flex-start; background:{accent_color}08;">'
    
    # Image column — resolve path using label-based prefix
    img_path = case.get('image_path', '')
    resolved = _resolve_case_image(img_path, case)
    if resolved and Path(resolved).exists():
        try:
            with open(resolved, 'rb') as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            ext = Path(resolved).suffix.lower().lstrip('.')
            mime = 'jpeg' if ext in ('jpg', 'jpeg') else ext
            card += f'<div style="flex-shrink:0;"><img src="data:image/{mime};base64,{img_b64}" style="width:180px; height:180px; object-fit:cover; border-radius:8px; border:1px solid #333;"/></div>'
        except Exception:
            pass
    
    # Text column
    card += '<div style="flex:1; min-width:0;">'
    card += f'<strong style="color:{accent_color};">{title}</strong>'
    if case.get('similarity_score'):
        card += f' <span style="color:#888; font-size:0.85rem;">— Similarity: {case["similarity_score"]:.3f}</span>'
    elif case.get('similarity'):
        card += f' <span style="color:#888; font-size:0.85rem;">— Similarity: {case["similarity"]:.3f}</span>'
    card += '<br/><br/>'
    
    if case.get('findings'):
        findings = str(case['findings'])[:350]
        card += f'<div style="font-size:0.9rem; line-height:1.5; color:#ddd; margin-bottom:0.5rem;">{findings}</div>'
    if case.get('impression'):
        imp = str(case['impression'])[:250]
        card += f'<div style="font-size:0.85rem; font-style:italic; color:#aaa;">Impression: {imp}</div>'
    
    card += '</div></div>'
    return card


def _resolve_case_image(img_path, case):
    """Resolve the stored relative image path to an absolute path on disk.
    
    MIMIC images are stored under:
      image-data/image/abnormal/files/p..   (label=1)
      image-data/image/healthy/files/p..    (label=0)
    Metadata stores relative paths starting with 'files/p...'.
    """
    if not img_path:
        return None
    
    # Try direct path first
    if Path(img_path).exists():
        return str(img_path)
    
    # Label-based resolution
    label = case.get('label', -1)
    if label == 1:
        candidate = Path('image-data/image/abnormal') / img_path
    elif label == 0:
        candidate = Path('image-data/image/healthy') / img_path
    else:
        # Try both
        candidate = Path('image-data/image/abnormal') / img_path
        if not candidate.exists():
            candidate = Path('image-data/image/healthy') / img_path
    
    if candidate.exists():
        return str(candidate)
    
    return None

# ============================================================================
# CUSTOM CSS
# ============================================================================

custom_css = """
/* Overall theme */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.app-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.app-header p {
    color: #888;
    font-size: 0.95rem;
}

/* Report box */
.report-box textarea {
    font-family: 'Georgia', serif !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    padding: 1.5rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    background: rgba(102, 126, 234, 0.03) !important;
}

/* Status badge */
.status-badge {
    text-align: center;
    padding: 0.5rem;
    font-weight: 600;
    border-radius: 8px;
}

/* Upload area */
.upload-area {
    border: 2px dashed rgba(102, 126, 234, 0.4) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease;
}
.upload-area:hover {
    border-color: rgba(102, 126, 234, 0.8) !important;
}

/* Smooth transitions */
.generating {
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
"""

# ============================================================================
# BUILD UI
# ============================================================================

def build_app():
    with gr.Blocks(css=custom_css, title="CXR Report Generator", theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    )) as demo:
        
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🩻 Chest X-ray Report Generator</h1>
            <p>AI-Powered Multi-Label Classification & Impression Generation</p>
            <p style="font-size: 0.8rem; color: #aaa; margin-top: 0.5rem;">
                BiomedCLIP · Contrastive Re-Ranking · Llama-3.1-8B
            </p>
        </div>
        """)
        
        with gr.Row(equal_height=False):
            # Left column: Upload
            with gr.Column(scale=1, min_width=350):
                image_input = gr.Image(
                    type="filepath",
                    label="Upload Chest X-ray",
                    height=380,
                    elem_classes=["upload-area"],
                    sources=["upload"],
                )
                
                analyze_btn = gr.Button(
                    "🔍 Generate Report",
                    variant="primary",
                    size="lg",
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    elem_classes=["status-badge"],
                )
                
                gr.Markdown("""
                ---
                **How it works:**
                1. Upload a frontal chest X-ray (PA/AP view)  
                2. Click **Generate Report**  
                3. The system embeds, classifies, retrieves similar cases, and generates a structured report
                
                *⏱️ Takes ~10-15 seconds per image*
                """)
            
            # Right column: Results
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 Generated Report"):
                        report_output = gr.Textbox(
                            label="Radiology Report",
                            lines=14,
                            interactive=False,
                            elem_classes=["report-box"],
                        )
                    
                    with gr.Tab("🏥 Assessment"):
                        assessment_output = gr.Markdown(
                            label="Clinical Assessment",
                        )
                    
                    with gr.Tab("🔬 Retrieved Evidence"):
                        evidence_output = gr.HTML(
                            label="Retrieved Evidence",
                        )
        
        # Wire up the button
        analyze_btn.click(
            fn=analyze_xray,
            inputs=[image_input],
            outputs=[report_output, assessment_output, evidence_output, status_output],
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 1rem 0; color: #888; font-size: 0.8rem;">
            <p>⚕️ For research and educational purposes only. Not for clinical use.</p>
            <p>Powered by BiomedCLIP · FAISS · Contrastive Re-Ranking · Llama-3.1-8B</p>
        </div>
        """)
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Pre-load pipeline
    print("=" * 60)
    print("  CXR Report Generator — Loading Pipeline...")
    print("=" * 60)
    load_pipeline()
    
    # Launch UI
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
