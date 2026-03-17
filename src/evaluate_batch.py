import argparse
import pandas as pd
import json
import random
from pathlib import Path
from pipeline_standalone import MedicalImagingPipeline
from tqdm import tqdm
import collections
import math
import re

# NLTK for BLEU-4 and METEOR
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Note: nltk not installed. BLEU-4 and METEOR unavailable.")

# Try to import rouge-score, fall back to simple implementation
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Note: rouge-score not installed. Using simple ROUGE approximation.")

# ============================================================================
# METRICS UTILS
# ============================================================================

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]

# Disease-only labels (exclude 'No Finding' for label overlap)
DISEASE_LABELS = [l for l in LABELS if l != 'No Finding']

# Global lookup: study_id -> set of positive CheXpert labels
STUDY_LABELS_LOOKUP = {}

# Simple keyword mapping for "poor man's CheXbert"
KEYWORD_MAPPING = {
    'Atelectasis': ['atelectasis', 'atelectatic', 'collapse'],
    'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'heart size'],
    'Consolidation': ['consolidation', 'consolidative', 'airspace disease'],
    'Edema': ['edema', 'congestion', 'fluid overload', 'vascular engorgement'],
    'Enlarged Cardiomediastinum': ['enlarged mediastinum', 'mediastinal widening', 'wide mediastinum'],
    'Fracture': ['fracture', 'broken rib', 'deformity'],
    'Lung Lesion': ['lesion', 'nodule', 'mass', 'granuloma'],
    'Lung Opacity': ['opacity', 'opacities', 'infiltrate', 'haze'],
    'Pleural Effusion': ['effusion', 'fluid in pleural', 'blunting'],
    'Pleural Other': ['pleural thickening', 'fibrosis', 'scarring'],
    'Pneumonia': ['pneumonia', 'infection'],
    'Pneumothorax': ['pneumothorax'],
    'Support Devices': ['device', 'pacemaker', 'wire', 'catheter', 'tube', 'clip', 'hardware'],
    'No Finding': ['no acute', 'normal', 'clear lungs', 'unremarkable']
}

# Negation patterns for keyword extraction
NEGATION_PATTERNS = ['no ', 'no\n', 'not ', 'without ', 'absent ', 'resolved ', 'negative for ',
                      'deny ', 'denies ', 'ruled out ', 'free of ', 'clear of ']

def extract_labels_from_text(text):
    """Negation-aware keyword-based label extraction."""
    text_lower = text.lower()
    detected = set()
    
    for label, keywords in KEYWORD_MAPPING.items():
        for kw in keywords:
            pos = text_lower.find(kw)
            if pos == -1:
                continue
            
            # Check for negation in the 30-char window before the keyword
            window_start = max(0, pos - 30)
            prefix = text_lower[window_start:pos]
            
            is_negated = any(neg in prefix for neg in NEGATION_PATTERNS)
            
            if not is_negated:
                detected.add(label)
                break  # Found a non-negated match for this label
    
    return detected

def compute_chexpert_metrics(results):
    """Compute Precision, Recall, F1 against Ground Truth columns."""
    tp = 0
    fp = 0
    fn = 0
    
    for res in results:
        # Ground Truth Labels (from CSV columns)
        gt_labels = set(res.get('ground_truth_labels', []))
        
        # Predicted Labels (Extracted from Generated Report Impression/Findings)
        # Note: We use the text because we want to eval the REPORT, not just the pipeline classifier
        report_text = res.get('generated_report', '')
        # Only focus on IMPRESSION and FINDINGS sections if we can parse them, but full text is safer for now
        pred_labels = extract_labels_from_text(report_text)
        
        # Calculate intersection/diffs
        # We calculate macro or micro? Let's do Micro-averaged over all instances
        tp += len(gt_labels.intersection(pred_labels))
        fp += len(pred_labels - gt_labels)
        fn += len(gt_labels - pred_labels)

    try:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except:
        precision, recall, f1 = 0, 0, 0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

def compute_bleu_1(reference, hypothesis):
    """Compute BLEU-1 score (unigram precision)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    ref_counts = collections.Counter(ref_words)
    hyp_counts = collections.Counter(hyp_words)
    
    overlap = sum((hyp_counts & ref_counts).values())
    total = len(hyp_words)
    
    if total == 0: return 0.0
    precision = overlap / total
    return precision

def compute_bleu_4(reference, hypothesis):
    """Compute BLEU-4 score with smoothing and brevity penalty (via nltk)."""
    if not NLTK_AVAILABLE:
        return 0.0
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words or not hyp_words:
        return 0.0
    smoothie = SmoothingFunction().method1
    try:
        return sentence_bleu([ref_words], hyp_words, 
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smoothie)
    except:
        return 0.0

def compute_meteor(reference, hypothesis):
    """Compute METEOR score (via nltk)."""
    if not NLTK_AVAILABLE:
        return 0.0
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words or not hyp_words:
        return 0.0
    try:
        return meteor_score([ref_words], hyp_words)
    except:
        return 0.0

def compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L score (longest common subsequence)."""
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    
    # Simple LCS-based approximation if library not available
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if not ref_words or not hyp_words:
        return 0.0
    
    # Simple LCS length calculation
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def load_study_labels_lookup(csv_path='image-data/processed/mimic_master.csv'):
    """Load study_id -> set of positive CheXpert labels for retrieval evaluation."""
    global STUDY_LABELS_LOOKUP
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        sid = int(row['study_id'])
        labels = set()
        for label in LABELS:
            if label in row and row[label] == 1.0:
                labels.add(label)
        STUDY_LABELS_LOOKUP[sid] = labels
    print(f"Loaded label lookup for {len(STUDY_LABELS_LOOKUP)} studies.")


def compute_retrieval_metrics(results):
    """
    Compute retrieval-specific metrics:
    - recall_at_5: fraction of queries where at least 1 retrieved case shares a disease label
    - avg_label_overlap: average Jaccard similarity between query and retrieved case labels
    - abnormal_alignment: for abnormal queries, fraction of retrieved cases that are abnormal (and vice versa)
    """
    if not results:
        return {}
    
    recall_hits = 0
    total_jaccard = 0
    total_retrieved = 0
    
    # Abnormal alignment: only count cases from the ABNORMAL retrieval bucket
    # (not the healthy comparison cases added by stratification)
    abn_query_abn_retrieved = 0
    abn_query_abn_bucket_total = 0
    norm_query_norm_retrieved = 0
    norm_query_total_retrieved = 0
    
    for res in results:
        query_gt = set(res.get('ground_truth_labels', []))
        query_diseases = query_gt - {'No Finding'}  # Disease labels only
        is_query_abnormal = 'No Finding' not in query_gt
        
        # Collect all retrieved cases (both abnormal and healthy buckets)
        retrieved = res.get('retrieved_cases', {})
        if isinstance(retrieved, dict):
            abn_bucket = retrieved.get('abnormal', [])
            healthy_bucket = retrieved.get('healthy', [])
            all_cases = abn_bucket + healthy_bucket
        elif isinstance(retrieved, list):
            all_cases = retrieved
            abn_bucket = retrieved
            healthy_bucket = []
        else:
            all_cases = []
            abn_bucket = []
            healthy_bucket = []
        
        if not all_cases:
            continue
        
        # Check each retrieved case for label overlap (Recall@5 and Jaccard)
        has_label_hit = False
        case_jaccards = []
        
        for case in all_cases:
            case_sid = case.get('study_id')
            if case_sid and int(case_sid) in STUDY_LABELS_LOOKUP:
                case_labels = STUDY_LABELS_LOOKUP[int(case_sid)]
                case_diseases = case_labels - {'No Finding'}
                
                # Label overlap (disease-level)
                if query_diseases and case_diseases:
                    intersection = query_diseases & case_diseases
                    union = query_diseases | case_diseases
                    jaccard = len(intersection) / len(union) if union else 0
                    case_jaccards.append(jaccard)
                    if intersection:
                        has_label_hit = True
                elif not query_diseases and not case_diseases:
                    # Both normal -> perfect match
                    case_jaccards.append(1.0)
                    has_label_hit = True
                else:
                    case_jaccards.append(0.0)
        
        # Abnormal alignment: within the ABNORMAL bucket only
        for case in abn_bucket:
            case_label = case.get('label', 0)
            if is_query_abnormal:
                abn_query_abn_bucket_total += 1
                if case_label == 1:
                    abn_query_abn_retrieved += 1
        
        # Normal alignment: within the HEALTHY bucket only (if stratified)
        # If no healthy bucket, use all cases for normal queries
        norm_bucket = healthy_bucket if healthy_bucket else all_cases
        for case in norm_bucket:
            case_label = case.get('label', 0)
            if not is_query_abnormal:
                norm_query_total_retrieved += 1
                if case_label == 0:
                    norm_query_norm_retrieved += 1
        
        if has_label_hit:
            recall_hits += 1
        if case_jaccards:
            total_jaccard += sum(case_jaccards) / len(case_jaccards)
            total_retrieved += 1
    
    count = len(results)
    metrics = {
        'recall_at_5': recall_hits / count if count > 0 else 0,
        'avg_label_overlap': total_jaccard / total_retrieved if total_retrieved > 0 else 0,
    }
    
    # Abnormal alignment (within abnormal bucket)
    if abn_query_abn_bucket_total > 0:
        metrics['abn_alignment'] = abn_query_abn_retrieved / abn_query_abn_bucket_total
    if norm_query_total_retrieved > 0:
        metrics['norm_alignment'] = norm_query_norm_retrieved / norm_query_total_retrieved
    
    return metrics


def compute_metrics(results):
    """Compute average metrics for the batch."""
    total_bleu1 = 0
    total_bleu4 = 0
    total_meteor = 0
    total_rouge = 0
    count = len(results)
    
    if count == 0: return {}
    
    # 1. NLG Scores
    for res in results:
        gt_text = (str(res.get('ground_truth_findings', '')) + " " + 
                  str(res.get('ground_truth_impression', ''))).strip()
        gen_text = res.get('generated_report', '')
        total_bleu1 += compute_bleu_1(gt_text, gen_text)
        total_bleu4 += compute_bleu_4(gt_text, gen_text)
        total_meteor += compute_meteor(gt_text, gen_text)
        total_rouge += compute_rouge_l(gt_text, gen_text)
        
    # 2. CheXpert Label Metrics
    chexpert_metrics = compute_chexpert_metrics(results)
    
    # 3. Retrieval Metrics
    retrieval_metrics = compute_retrieval_metrics(results)
    
    metrics = {
        "avg_bleu_1": total_bleu1 / count,
        "avg_bleu_4": total_bleu4 / count,
        "avg_meteor": total_meteor / count,
        "avg_rouge_l": total_rouge / count,
        "sample_size": count
    }
    metrics.update(chexpert_metrics)
    metrics.update(retrieval_metrics)
    return metrics

# ============================================================================
# DATA LOADING
# ============================================================================

def get_image_full_path(relative_path, base_dir='image-data'):
    """Resolve full image path checking both healthy and abnormal directories."""
    abnormal_path = Path(base_dir) / 'image' / 'abnormal' / relative_path
    if abnormal_path.exists():
        return str(abnormal_path)
    healthy_path = Path(base_dir) / 'image' / 'healthy' / relative_path
    if healthy_path.exists():
        return str(healthy_path)
    return None

def load_test_data(csv_path, sample_size=None, seed=42):
    """Load test data and sample N items with reproducible seed."""
    print(f"Loading test data from {csv_path} (seed={seed})...")
    df = pd.read_csv(csv_path)
    
    valid_data = []
    print("Verifying image paths...")
    
    indices = list(range(len(df)))
    if sample_size and sample_size < len(df):
        random.seed(seed)  # Reproducible sampling
        random.shuffle(indices)
        # Note: We iterate through ALL shuffled indices until we find enough valid images
    
    count = 0
    for idx in indices:
        row = df.iloc[idx]
        full_path = get_image_full_path(row['image_path'])
        
        if full_path:
            # Extract GT labels where value is 1.0
            gt_labels = []
            for label in LABELS:
                if label in row and row[label] == 1.0:
                    gt_labels.append(label)
            
            valid_data.append({
                'image_path': full_path,
                'findings': row.get('findings', ''),
                'impression': row.get('impression', ''),
                'subject_id': row.get('subject_id'),
                'ground_truth_labels': gt_labels
            })
            count += 1
            if sample_size and count >= sample_size:
                break
                
    print(f"Found {len(valid_data)} validated images.")
    return valid_data

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical Imaging Pipeline")
    parser.add_argument('--csv', default='image-data/processed/test.csv', help='Path to test CSV')
    parser.add_argument('--sample_size', type=int, default=20, help='Number of images to evaluate (Default: 20)')
    parser.add_argument('--output', default='results/batch_evaluation.json', help='Output JSON path')
    parser.add_argument('--ablation', default='full', 
                        choices=['full', 'cosine_only', 'stratify_only', 'contrastive_only', 'no_retrieval'],
                        help='Ablation mode: full (proposed), cosine_only, stratify_only, contrastive_only, no_retrieval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    parser.add_argument('--llm', default='ollama', choices=['ollama', 'openai'],
                        help='LLM backend: ollama (llama3.1:8b) or openai (gpt-4o-mini)')
    args = parser.parse_args()
    
    print(f"Ablation mode: {args.ablation}")
    
    # Load study labels for retrieval metrics
    load_study_labels_lookup()
    
    # 1. Load Data
    data = load_test_data(args.csv, args.sample_size, seed=args.seed)
    if not data:
        print("No valid data found. Exiting.")
        return

    # 2. Initialize Pipeline
    pipeline = MedicalImagingPipeline(llm_provider=args.llm)
    llm_model = pipeline.llm_model_name
    
    # 3. Run Batch
    results = []
    print(f"\nRunning pipeline on {len(data)} images (ablation={args.ablation})...")
    
    for item in tqdm(data):
        try:
            # Run prediction with ablation mode
            res = pipeline.predict(item['image_path'], retrieve_k=5, ablation_mode=args.ablation)
            
            # Enrich result
            res['ground_truth_findings'] = item['findings']
            res['ground_truth_impression'] = item['impression']
            res['subject_id'] = int(item['subject_id']) if pd.notnull(item['subject_id']) else None
            res['ground_truth_labels'] = item['ground_truth_labels']
            
            results.append(res)
            
        except Exception as e:
            print(f"Error processing {item['image_path']}: {e}")
            
    # 4. Compute Metrics
    print("\nComputing metrics...")
    
    # Split results by Ground Truth
    # Normal = 'No Finding' is in GT labels
    normal_results = [r for r in results if 'No Finding' in r.get('ground_truth_labels', [])]
    # Abnormal = 'No Finding' is NOT in GT labels (and labeled with something else)
    abnormal_results = [r for r in results if 'No Finding' not in r.get('ground_truth_labels', [])]
    
    metrics_overall = compute_metrics(results)
    metrics_normal = compute_metrics(normal_results)
    metrics_abnormal = compute_metrics(abnormal_results)
    
    # 5. Save Results
    output_data = {
        'config': {
            'ablation_mode': args.ablation,
            'sample_size': args.sample_size,
            'seed': args.seed,
            'csv': args.csv,
            'llm': llm_model,
            'llm_provider': args.llm,
        },
        'metrics_overall': metrics_overall,
        'metrics_normal': metrics_normal,
        'metrics_abnormal': metrics_abnormal,
        'results': results
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print("="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    def print_metrics(name, m):
        print(f"\n--- {name} (N={m.get('sample_size', 0)}) ---")
        print(f"Avg BLEU-1:       {m.get('avg_bleu_1', 0):.4f}")
        print(f"Avg BLEU-4:       {m.get('avg_bleu_4', 0):.4f}")
        print(f"Avg METEOR:       {m.get('avg_meteor', 0):.4f}")
        print(f"Avg ROUGE-L:      {m.get('avg_rouge_l', 0):.4f}")
        print(f"Precision:        {m.get('precision', 0):.4f}")
        print(f"Recall:           {m.get('recall', 0):.4f}")
        print(f"F1 Score:         {m.get('f1', 0):.4f}")
        print(f"Recall@5:         {m.get('recall_at_5', 0):.4f}")
        print(f"Avg Label Overlap:{m.get('avg_label_overlap', 0):.4f}")
        if 'abn_alignment' in m:
            print(f"Abn Alignment:    {m.get('abn_alignment', 0):.4f}")
        if 'norm_alignment' in m:
            print(f"Norm Alignment:   {m.get('norm_alignment', 0):.4f}")

    print_metrics("OVERALL", metrics_overall)
    print_metrics("NORMAL Cases", metrics_normal)
    print_metrics("ABNORMAL Cases", metrics_abnormal)
    
    print("="*60)
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
