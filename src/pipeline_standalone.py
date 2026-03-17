"""
Complete Medical Imaging Pipeline - Standalone Script

This script provides a self-contained pipeline that doesn't require running other notebooks.
Use this if you encounter issues with %run magic in Jupyter.
"""

import torch
import torch.nn as nn
import open_clip
from PIL import Image
import faiss
import pickle
import faiss
import pickle
import json
import subprocess
from pathlib import Path

# LangChain imports
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env for OpenAI API key (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# FROZEN PROMPT (identical for all LLM backends)
# ============================================================================

REPORT_SYSTEM_PROMPT_V1 = (
    "You are a radiologist assistant. Generate a structured chest X-ray report.\n\n"
    "CRITICAL RULES:\n"
    "1. Only mention conditions that are EXPLICITLY SUPPORTED by the classifier predictions or retrieved evidence.\n"
    "2. Do NOT invent, speculate, or hallucinate about conditions not present in the evidence.\n"
    "3. If a condition has classifier probability < 0.5, do NOT mention it.\n"
    "4. Do not attribute etiology unless explicitly stated in retrieved evidence.\n"
    "5. If evidence is insufficient, explicitly state limitations.\n"
    "6. Use cautious clinical language (e.g., 'suggests', 'consistent with', 'may indicate').\n"
)

REPORT_TASK_INSTRUCTION_V1 = (
    "\nTASK:\n"
    "Generate a concise medical report with:\n"
    "1. FINDINGS: Describe observations based on the query and similar abnormal cases. "
    "Only mention conditions with classifier confidence > 0.5.\n"
    "2. COMPARISON: Compare with retrieved cases (especially healthy vs abnormal patterns). "
    "Explicitly state at least one feature present in abnormal cases but absent in healthy references.\n"
    "3. IMPRESSION: Radiographic impression summarizing likely patterns, not definitive diagnosis.\n\n"
    "IMPORTANT: Do NOT mention any condition that is not directly supported by the provided evidence. "
    "If you are unsure, say 'insufficient evidence' rather than speculating.\n\n"
    "If limitations exist, include one sentence starting with \"Limitations:\"\n"
)


# ============================================================================
# COMPONENT CLASSES
# ============================================================================

class BiomedCLIPEmbedder:
    """BiomedCLIP embedding generator."""
    
    def __init__(self, model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'):
        print(f"Loading {model_name}...")
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
    def encode_image(self, image_path):
        """Generate embedding for a single image."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emb = self.model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        return emb.cpu()


class MultiLabelClassifier(nn.Module):
    """Multi-label disease classifier."""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def load_classifier(checkpoint_path='models/multilabel_classifier_biomedclip_v2.pt', device='cpu'):
    """Load trained classifier. Returns (model, temperature)."""
    model = MultiLabelClassifier()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle nested checkpoint structure
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        temperature = checkpoint.get('temperature', 1.0)
    else:
        model.load_state_dict(checkpoint)
        temperature = 1.0
    
    model.eval()
    model.to(device)
    return model, temperature


def predict(model, embedding, device='cpu', temperature=1.0):
    """Predict disease probabilities with temperature scaling."""
    embedding = embedding.to(device)
    with torch.no_grad():
        logits = model(embedding)
        probs = torch.sigmoid(logits / temperature)
    return probs.cpu()


class AbnormalityDetector:
    """Abnormality detection using healthy centroid."""
    
    def __init__(self, centroid_path='embeddings/mimic_healthy_centroid.pt'):
        self.centroid = torch.load(centroid_path, weights_only=False)
        self.thresholds = {
            'healthy_90': 0.15,
            'healthy_95': 0.20
        }
    
    def compute_score(self, embedding):
        """Compute abnormality score."""
        cos_sim = torch.cosine_similarity(embedding, self.centroid, dim=-1)
        return (1.0 - cos_sim).item()
    
    def classify(self, score):
        """Classify abnormality level."""
        if score < self.thresholds['healthy_90']:
            return "normal"
        elif score < self.thresholds['healthy_95']:
            return "borderline"
        else:
            return "abnormal"
    
    def analyze(self, embedding):
        """Full analysis: score + classification."""
        score = self.compute_score(embedding)
        label = self.classify(score)
        return {
            'abnormality_score': score,
            'abnormality_label': label
        }


class MedicalImageRetriever:
    """FAISS-based medical image retrieval."""
    
    def __init__(self, 
                 index_path='faiss_index/mimic_image_index_cpu.faiss',
                 metadata_path='faiss_index/mimic_faiss_metadata.pkl',
                 contrastive_params=None):
        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        
        print("Loading metadata...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.contrastive_params = contrastive_params or {'lambda': 0.5, 'mu': 0.3}
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def retrieve(self, query_embedding, k=10, stratify=True, abnormality_threshold=0.5, query_abnormality_score=0.0):
        """
        Retrieve similar medical images with contrastive re-ranking.
        
        Formula: final_score = similarity * (1 + lambda * case_abn) * (1 + mu * query_abn)
        """
        # Convert to numpy
        query_np = query_embedding.numpy().astype('float32')
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        # Search FAISS index
        search_k = k * 4 if stratify else k
        scores, indices = self.index.search(query_np, k=search_k)
        
        # Constants for re-ranking (from config or defaults)
        LAMBDA = self.contrastive_params.get('lambda', 0.5)
        MU = self.contrastive_params.get('mu', 0.3)
        
        # Normalize query abnormality score (0.3 is approx max distance usually observed)
        norm_query_abn = min(query_abnormality_score / 0.3, 1.0)
        
        # 1. Collect and Re-rank Candidates
        candidates = []
        for idx, score in zip(indices[0], scores[0]):
            case = self.metadata[idx].copy()
            
            # Base similarity (Cosine)
            similarity = float(score)
            
            # Get case abnormality score (default to 0 if missing)
            case_abn = case.get('abnormality_score', 0.0)
            
            # Suppression: If case is healthy (below threshold), do not boost
            if case_abn < abnormality_threshold:
                case_abn = 0.0
            else:
                # Normalize case abnormality
                case_abn = min(case_abn / 0.3, 1.0)
            
            # Compute Contrastive Score
            # final_score = similarity * (1 +  * case_abn) * (1 +  * query_abn)
            contrastive_score = similarity * (1 + LAMBDA * case_abn) * (1 + MU * norm_query_abn)
            
            case['similarity'] = contrastive_score
            case['original_similarity'] = similarity
            
            candidates.append(case)
            
        # Sort by new contrastive score
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        if not stratify:
            return candidates[:k]
        
        # 2. Stratify by abnormality (using re-ranked order)
        abnormal_cases = []
        healthy_cases = []
        
        for case in candidates:
            # Determine if case is abnormal for stratification bucket
            # Use ground truth label if available, otherwise use score
            if 'label' in case:
                is_abnormal = (case['label'] == 1)
            else:
                abn_score = case.get('abnormality_score', 0)
                is_abnormal = (abn_score > abnormality_threshold)
            
            if is_abnormal:
                abnormal_cases.append(case)
            else:
                healthy_cases.append(case)
            
            if len(abnormal_cases) >= k and len(healthy_cases) >= k:
                break
        
        return {
            'abnormal': abnormal_cases[:k],
            'healthy': healthy_cases[:k]
        }
    
    def _format_results(self, indices, scores):
        """Deprecated: Internal formatting moved to retrieve loop."""
        pass




# ============================================================================
# REPORT GENERATION COMPONENTS (LANGCHAIN)
# ============================================================================

def safe_snippet(text, max_sentences=3):
    """Safely extract first N sentences to avoid cutting mid-sentence."""
    if not isinstance(text, str):
        return ""
    # Simple split by period, could be made more robust with nltk/spacy
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return '. '.join(sentences[:max_sentences]) + '.'

def build_report_context(abnormality_result, disease_probs, retrieved_cases):
    """Build context dictionaries for LangChain."""
    
    # 1. Query Context
    query_context = f"System Classification: {abnormality_result['abnormality_label'].upper()}\n"
    query_context += f"Abnormality Score: {abnormality_result['abnormality_score']:.3f}\n"
    
    # Add explicit override warning if present
    if abnormality_result.get('override_reason'):
        query_context += f"NOTE: Query flagged ABNORMAL due to high classifier confidence ({abnormality_result['override_reason']}) despite low geometric score.\n"
    
    query_context += "Predicted Abnormalities (classifier confidence):\n"
    
    # Filter diseases: ONLY include those with prob >= 0.5 (calibrated threshold)
    filtered_diseases = {k: v for k, v in disease_probs.items() if v >= 0.5}
    
    # Sort filtered diseases
    top_diseases = sorted(filtered_diseases.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Check if we have significant findings (non-No Finding diseases)
    has_real_findings = any(disease != 'No Finding' for disease, prob in top_diseases)
    
    # Build the query context with only high-confidence predictions
    diseases_added = 0
    for disease, prob in top_diseases:
        # Suppress 'No Finding' if we have other findings
        if disease == 'No Finding' and has_real_findings:
            continue
        query_context += f"- {disease} ({prob:.2f})\n"
        diseases_added += 1
    
    # If no diseases passed the threshold, indicate insufficient evidence
    if diseases_added == 0:
        query_context += "- No conditions met the confidence threshold (>= 0.5). Report insufficient evidence.\n"

    # 2. Evidence Context (Retrieved cases)
    # CRITICAL FIX: If system says NORMAL, do NOT show abnormal retrieved cases to prevent hallucination
    evidence_text = ""
    is_normal = (abnormality_result['abnormality_label'] == 'normal')
    
    if not is_normal:
        # Add abnormal cases only if we think the image is abnormal
        for i, case in enumerate(retrieved_cases.get('abnormal', [])[:3], 1):
            evidence_text += f"\n[Abnormal Case {i} (Contrastive Score: {case.get('similarity', 0):.2f})]\n"
            if 'findings' in case:
                evidence_text += f"Findings: {safe_snippet(case['findings'])}\n"
            elif 'impression' in case:
                 evidence_text += f"Impression: {safe_snippet(case['impression'])}\n"
    else:
        evidence_text += "\n(System Classification is NORMAL - Abnormal retrieved cases suppressed to prevent hallucination)\n"

    # Add healthy references
    for i, case in enumerate(retrieved_cases.get('healthy', [])[:2], 1):
        evidence_text += f"\n[Healthy Reference {i}]\n"
        if 'impression' in case:
            evidence_text += f"Impression: {safe_snippet(case['impression'])}\n"
            
    return {
        "query_data": query_context,
        "retrieved_evidence": evidence_text
    }

class ReportGenerator:
    def __init__(self, model_name="llama3.1:8b", config=None, llm_provider="ollama"):
        self.config = config
        self.llm_provider = llm_provider
        self.model_name = model_name
        
        # Initialize LangChain Chat Model based on provider
        if llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            import os
            print(f"   Initializing LangChain ChatOpenAI ({model_name})...")
            self.llm = ChatOpenAI(model=model_name, temperature=0.2,
                                 api_key=os.getenv('OPENAI_API_KEY'))
        else:
            print(f"   Initializing LangChain ChatOllama ({model_name})...")
            self.llm = ChatOllama(model=model_name, temperature=0.2)
        
        # Use frozen prompt constants (identical across all LLM backends)
        system_template = REPORT_SYSTEM_PROMPT_V1
        task_instruction = REPORT_TASK_INSTRUCTION_V1
            
        # Create ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", f"""QUERY DATA:
{{query_data}}

RETRIEVED EXAMPLES:
{{retrieved_evidence}}

{task_instruction}
""")
        ])
        
        # Create Chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def generate(self, abnormality_result, disease_probs, retrieved_cases):
        # Prepare inputs
        inputs = build_report_context(abnormality_result, disease_probs, retrieved_cases)
        
        print("   Generating report with LangChain...")
        try:
            return self.chain.invoke(inputs)
        except Exception as e:
            print(f" LangChain Generation Error: {e}")
            return f"Error generating report: {e}"


class LLMJudge:
    """LLM-as-Judge for factuality checking of generated reports."""
    
    def __init__(self, model_name="llama3.1:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical report quality evaluator. 
Rate the factuality of generated radiology reports on a scale of 1-5.

SCORING:
5 = Fully grounded in evidence, no unsupported claims
4 = Mostly grounded, minor speculation clearly marked
3 = Some unsupported claims but core findings accurate
2 = Several unsupported claims or inaccuracies  
1 = Major hallucinations or contradicts evidence

Be strict. If the report mentions conditions not supported by the classifier predictions or retrieved cases, deduct points."""),
            ("user", """CLASSIFIER PREDICTIONS:
{predictions}

RETRIEVED EVIDENCE:
{evidence}

GENERATED REPORT:
{report}

Rate this report's factuality (1-5) and explain briefly.
Format: SCORE: [1-5]
REASON: [brief explanation]""")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def evaluate(self, report, disease_probs, retrieved_cases):
        """Evaluate report factuality. Returns (score, reasoning)."""
        try:
            # Format predictions
            top_probs = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            pred_text = "\n".join([f"- {d}: {p:.2f}" for d, p in top_probs])
            
            # Format evidence
            evidence_text = ""
            for case in retrieved_cases.get('abnormal', [])[:2]:
                if case.get('findings'):
                    evidence_text += f"[Abnormal] {case['findings'][:150]}...\n"
            for case in retrieved_cases.get('healthy', [])[:1]:
                if case.get('impression'):
                    evidence_text += f"[Healthy] {case['impression'][:100]}...\n"
            
            if not evidence_text:
                evidence_text = "(No text evidence available)"
            
            result = self.chain.invoke({
                "predictions": pred_text,
                "evidence": evidence_text,
                "report": report
            })
            
            # Parse score
            score = 3  # default
            if "SCORE:" in result:
                try:
                    score_str = result.split("SCORE:")[1].split()[0].strip()
                    score = int(score_str)
                except:
                    pass
            
            return score, result
            
        except Exception as e:
            return 3, f"Judge error: {e}"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MedicalImagingPipeline:
    """Complete end-to-end medical imaging pipeline."""
    
    def __init__(self, llm_provider="ollama"):
        print(" Initializing Medical Imaging Pipeline...")
        
        # Load all components
        # Load config if present
        self.config = {}
        if Path('pipeline_config.json').exists():
            with open('pipeline_config.json', 'r') as f:
                self.config = json.load(f)
            print("  Loaded configuration from pipeline_config.json")
        
        print("  Loading embedder...")
        self.embedder = BiomedCLIPEmbedder()
        
        print("  Loading classifier...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier, self.temperature = load_classifier(device=self.device)
        
        print("  Loading abnormality detector...")
        self.abnormality_detector = AbnormalityDetector()
        
        print("  Loading retriever...")
        retrieval_config = self.config.get('retrieval', {})
        contrastive_params = {
            'lambda': retrieval_config.get('contrastive_lambda', 0.5),
            'mu': retrieval_config.get('contrastive_mu', 0.3)
        }
        self.retriever = MedicalImageRetriever(contrastive_params=contrastive_params)
        
        print("  Loading LLM...")
        if llm_provider == "openai":
            model_name = "gpt-4o-mini"
        else:
            model_name = self.config.get('llm', {}).get('model', "llama3.1:8b")
        self.llm_model_name = model_name
        self.llm_provider = llm_provider
        self.reporter = ReportGenerator(model_name=model_name, config=self.config, llm_provider=llm_provider)

        
        # CheXpert labels
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
        ]
        
        print(" Pipeline ready!\n")
    
    def predict(self, image_path, retrieve_k=5, ablation_mode='full'):
        """
        Run full pipeline on an image.
        
        Args:
            image_path: Path to chest X-ray image
            retrieve_k: Number of similar cases to retrieve per category
            ablation_mode: One of:
                'full'             - Contrastive re-ranking + stratification (proposed method)
                'cosine_only'      - Plain cosine similarity, no stratification
                'stratify_only'    - Cosine + stratification, no contrastive boost
                'contrastive_only' - Contrastive re-ranking, no stratification
                'no_retrieval'     - No retrieval at all, classifier-only baseline
        
        Returns:
            dict: Complete analysis results
        """
        print(f" Analyzing: {image_path}")
        
        # 1. Generate embedding
        print("  [1/4] Generating embedding...")
        embedding = self.embedder.encode_image(image_path)
        
        # 2. Abnormality detection
        print("  [2/4] Detecting abnormality...")
        abnormality = self.abnormality_detector.analyze(embedding)
        
        # 3. Multi-label classification (Override abnormality if disease detected)
        print("  [3/4] Classifying diseases...")
        probs = predict(self.classifier, embedding, device=self.device, temperature=self.temperature)
        disease_probs = {
            label: float(prob) 
            for label, prob in zip(self.labels, probs[0])
        }
        
        abnormality['override_reason'] = None
        
        # Check if classifier detected anything (threshold 0.6 to reduce false positives)
        # FIX: Ignore 'No Finding' when checking for high-confidence disease predictions
        relevant_probs = {k: v for k, v in disease_probs.items() if k != 'No Finding'}
        max_prob = max(relevant_probs.values()) if relevant_probs else 0.0
        
        if max_prob > 0.60 and abnormality['abnormality_label'] == 'normal':
            print(f"   Classifier detected disease (prob {max_prob:.2f}) -> Flagging as ABNORMAL")
            abnormality['abnormality_label'] = 'abnormal'
            abnormality['override_reason'] = 'classifier_confidence'
        
        # 4. Retrieve similar cases
        print(f"  [4/4] Retrieving similar cases (mode={ablation_mode})...")
        
        if ablation_mode == 'no_retrieval':
            # Baseline: no retrieval at all
            retrieved = {'abnormal': [], 'healthy': []}
        else:
            # Determine contrastive params based on ablation mode
            if ablation_mode in ('cosine_only', 'stratify_only'):
                # Disable contrastive boost: lambda=0, mu=0
                saved_params = self.retriever.contrastive_params.copy()
                self.retriever.contrastive_params = {'lambda': 0.0, 'mu': 0.0}
            
            # Determine stratification
            use_stratify = ablation_mode in ('full', 'stratify_only')
            
            retrieved = self.retriever.retrieve(
                embedding, 
                k=retrieve_k, 
                stratify=use_stratify,
                abnormality_threshold=0.5,
                query_abnormality_score=abnormality['abnormality_score']
            )
            
            # Restore original params if we overrode them
            if ablation_mode in ('cosine_only', 'stratify_only'):
                self.retriever.contrastive_params = saved_params
            
            # If not stratified, wrap in dict format for consistency
            if not use_stratify:
                retrieved = {'abnormal': retrieved, 'healthy': []}
        
        # 5. Generate Report
        print("  [5/5] Generating Medical Report...")
        
        # Optimize VRAM: Move PyTorch models to CPU to free space for Ollama
        if self.device == 'cuda':
            print("   Offloading models to CPU to free VRAM for LLM...")
            self.embedder.model.cpu()
            self.classifier.cpu()
            torch.cuda.empty_cache()
            
        report_text = self.reporter.generate(
            abnormality,
            disease_probs,
            retrieved
        )
        
        # Move models back to GPU for subsequent predictions (batch processing)
        if self.device == 'cuda':
            self.embedder.model.cuda()
            self.classifier.cuda()
            self.embedder.device = 'cuda'  # Update device tracker
        
        print(" Analysis complete!\n")
        
        return {
            'image_path': str(image_path),
            'abnormality_score': abnormality['abnormality_score'],
            'abnormality_label': abnormality['abnormality_label'],
            'override_reason': abnormality.get('override_reason'),
            'disease_probabilities': disease_probs,
            'top_diseases': sorted(
                disease_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'retrieved_cases': retrieved,
            'generated_report': report_text
        }
    
    def print_results(self, results):
        """Pretty print results."""
        print("="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n Image: {results['image_path']}")
        
        print(f"\n Abnormality Score: {results['abnormality_score']:.4f}")
        print(f"   Classification: {results['abnormality_label'].upper()}")
        
        print("\n Top 5 Predicted Diseases:")
        for i, (disease, prob) in enumerate(results['top_diseases'], 1):
            bar = '' * int(prob * 20)
            print(f"   {i}. {disease:30s} {prob:.3f} {bar}")
        
        print(f"\n Retrieved Similar Cases:")
        print(f"   Abnormal: {len(results['retrieved_cases']['abnormal'])} cases")
        for i, case in enumerate(results['retrieved_cases']['abnormal'], 1):
             print(f"     - Case {i}: Score {case.get('similarity',0):.3f} (Cos: {case.get('original_similarity',0):.3f})")
        print(f"   Healthy:  {len(results['retrieved_cases']['healthy'])} cases")
        
        print(f"\n Generated Report:")
        print("-" * 30)
        print(results['generated_report'])
        print("-" * 30)
        
        print("\n" + "="*60)
    
    def save_results(self, results, output_path):
        """Save results to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f" Results saved to {output_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MedicalImagingPipeline()
    
    # Example: Analyze a test image
    test_image = './image-data/image/abnormal/files/p10/p10000935/s51178377/9b314ad7-fbcb0422-6db62dfc-732858d0-a5527d8b.jpg'
    results = pipeline.predict(test_image, retrieve_k=5)
    pipeline.print_results(results)
    pipeline.save_results(results, 'results/test_case.json')
    
    print("Pipeline initialized and ready to use!")
    print("\nExample usage:")
    print("  results = pipeline.predict('path/to/xray.jpg')")
    print("  pipeline.print_results(results)")
