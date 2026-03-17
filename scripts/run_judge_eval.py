"""
LLM-as-Judge: Final Qualitative Evaluation
- Generator: GPT-4o-mini (already run)
- Judge: GPT-4o
- N = 12 representative cases (mix of normal + abnormal)
- Compares: cosine_only vs full
- Scores: Clinical Correctness, Hallucination, Evidence Grounding (0-5)
"""

import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load existing GPT-4o-mini results
cosine_data = json.load(open('results/gpt4o_cosine_only.json'))
full_data = json.load(open('results/gpt4o_full.json'))

cosine_results = cosine_data['results']
full_results = full_data['results']

# Select 12 representative cases: 6 normal + 6 abnormal
normal_indices = [i for i, r in enumerate(cosine_results) 
                  if 'No Finding' in r.get('ground_truth_labels', [])]
abnormal_indices = [i for i, r in enumerate(cosine_results) 
                    if 'No Finding' not in r.get('ground_truth_labels', [])]

print(f"Available: {len(normal_indices)} normal, {len(abnormal_indices)} abnormal")

# Pick all 12 abnormal + first 18 normal = 30 total
selected = normal_indices[:18] + abnormal_indices[:12]
print(f"Selected {len(selected)} cases for judging (target: 30)")

# Initialize GPT-4o judge
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=os.getenv('OPENAI_API_KEY'))

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert radiologist evaluating AI-generated chest X-ray reports.

You will be given:
1. Ground truth labels (CheXpert disease labels from the study)
2. Classifier predictions (disease probabilities from the system)
3. Retrieved evidence (reports from similar cases used as context)
4. The generated report to evaluate

Score the report on THREE criteria, each from 0 to 5:

**Clinical Correctness (0-5):**
0 = Completely wrong findings
5 = All findings consistent with the evidence and ground truth

**Hallucination Score (0-5):**
0 = Many invented/unsupported findings
5 = No hallucinations, all claims grounded in evidence

**Evidence Grounding (0-5):**
0 = Report ignores retrieved evidence entirely
5 = Report clearly relies on and references retrieved evidence

IMPORTANT: Judge ONLY factual/clinical quality. Do NOT judge writing style, grammar, or formatting.

Respond in EXACTLY this format (nothing else):
CLINICAL_CORRECTNESS: [0-5]
HALLUCINATION: [0-5]
EVIDENCE_GROUNDING: [0-5]
BRIEF_NOTE: [One sentence observation]"""),
    ("user", """GROUND TRUTH LABELS: {gt_labels}

CLASSIFIER PREDICTIONS (top diseases):
{predictions}

RETRIEVED EVIDENCE:
{evidence}

GENERATED REPORT:
{report}

Score this report.""")
])

judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()

def format_predictions(result):
    """Format top disease predictions for judge."""
    top = result.get('top_diseases', [])
    if top:
        return "\n".join([f"- {d}: {p:.3f}" for d, p in top[:5]])
    return "(No predictions available)"

def format_evidence(result):
    """Format retrieved evidence for judge."""
    retrieved = result.get('retrieved_cases', {})
    evidence = ""
    for case in retrieved.get('abnormal', [])[:2]:
        if case.get('findings'):
            evidence += f"[Similar Abnormal Case] {str(case['findings'])[:200]}\n"
        if case.get('impression'):
            evidence += f"  Impression: {str(case['impression'])[:150]}\n"
    for case in retrieved.get('healthy', [])[:2]:
        if case.get('impression'):
            evidence += f"[Healthy Reference] {str(case['impression'])[:150]}\n"
    return evidence if evidence else "(No text evidence available)"

def parse_judge_response(response):
    """Parse the judge's structured response."""
    scores = {'clinical_correctness': 0, 'hallucination': 0, 'evidence_grounding': 0, 'note': ''}
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.startswith('CLINICAL_CORRECTNESS:'):
            try: scores['clinical_correctness'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('HALLUCINATION:'):
            try: scores['hallucination'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('EVIDENCE_GROUNDING:'):
            try: scores['evidence_grounding'] = int(line.split(':')[1].strip())
            except: pass
        elif line.startswith('BRIEF_NOTE:'):
            scores['note'] = line.split(':', 1)[1].strip()
    return scores

# Run judge on selected cases
judge_results = []

for idx in selected:
    cos_res = cosine_results[idx]
    full_res = full_results[idx]
    
    gt_labels = cos_res.get('ground_truth_labels', [])
    case_type = "NORMAL" if "No Finding" in gt_labels else "ABNORMAL"
    
    print(f"\nCase {idx} ({case_type}): GT={gt_labels}")
    
    # Judge cosine_only report
    print(f"  Judging cosine_only...")
    try:
        cos_response = judge_chain.invoke({
            'gt_labels': ', '.join(gt_labels),
            'predictions': format_predictions(cos_res),
            'evidence': format_evidence(cos_res),
            'report': cos_res.get('generated_report', '')
        })
        cos_scores = parse_judge_response(cos_response)
    except Exception as e:
        print(f"  Error: {e}")
        cos_scores = {'clinical_correctness': 0, 'hallucination': 0, 'evidence_grounding': 0, 'note': f'Error: {e}'}
    
    time.sleep(0.5)  # Rate limit
    
    # Judge full report
    print(f"  Judging full...")
    try:
        full_response = judge_chain.invoke({
            'gt_labels': ', '.join(gt_labels),
            'predictions': format_predictions(full_res),
            'evidence': format_evidence(full_res),
            'report': full_res.get('generated_report', '')
        })
        full_scores = parse_judge_response(full_response)
    except Exception as e:
        print(f"  Error: {e}")
        full_scores = {'clinical_correctness': 0, 'hallucination': 0, 'evidence_grounding': 0, 'note': f'Error: {e}'}
    
    time.sleep(0.5)  # Rate limit
    
    judge_results.append({
        'case_idx': idx,
        'case_type': case_type,
        'gt_labels': gt_labels,
        'cosine_scores': cos_scores,
        'full_scores': full_scores
    })
    
    print(f"  cosine: CC={cos_scores['clinical_correctness']} H={cos_scores['hallucination']} EG={cos_scores['evidence_grounding']}")
    print(f"  full:   CC={full_scores['clinical_correctness']} H={full_scores['hallucination']} EG={full_scores['evidence_grounding']}")

# Save raw results
with open('results/judge_results.json', 'w') as f:
    json.dump(judge_results, f, indent=2)

# Generate output table
lines = []
lines.append("LLM-AS-JUDGE EVALUATION")
lines.append(f"Generator: GPT-4o-mini | Judge: GPT-4o | N={len(judge_results)}")
lines.append("=" * 90)
lines.append("")

# Header
lines.append(f"{'Case':>6s} {'Type':>8s} {'Method':>14s} {'ClinCorr':>10s} {'Halluc':>8s} {'EvGround':>10s}")
lines.append("-" * 60)

cos_cc, cos_h, cos_eg = [], [], []
full_cc, full_h, full_eg = [], [], []

for jr in judge_results:
    cs = jr['cosine_scores']
    fs = jr['full_scores']
    
    lines.append(f"{jr['case_idx']:>6d} {jr['case_type']:>8s} {'cosine_only':>14s} {cs['clinical_correctness']:>10d} {cs['hallucination']:>8d} {cs['evidence_grounding']:>10d}")
    lines.append(f"{'':>6s} {'':>8s} {'full':>14s} {fs['clinical_correctness']:>10d} {fs['hallucination']:>8d} {fs['evidence_grounding']:>10d}")
    lines.append("")
    
    cos_cc.append(cs['clinical_correctness'])
    cos_h.append(cs['hallucination'])
    cos_eg.append(cs['evidence_grounding'])
    full_cc.append(fs['clinical_correctness'])
    full_h.append(fs['hallucination'])
    full_eg.append(fs['evidence_grounding'])

# Mean scores
lines.append("=" * 60)
lines.append("MEAN SCORES")
lines.append("-" * 60)
n = len(judge_results)
lines.append(f"{'':>6s} {'':>8s} {'cosine_only':>14s} {sum(cos_cc)/n:>10.2f} {sum(cos_h)/n:>8.2f} {sum(cos_eg)/n:>10.2f}")
lines.append(f"{'':>6s} {'':>8s} {'full':>14s} {sum(full_cc)/n:>10.2f} {sum(full_h)/n:>8.2f} {sum(full_eg)/n:>10.2f}")
lines.append(f"{'':>6s} {'':>8s} {'delta':>14s} {(sum(full_cc)-sum(cos_cc))/n:>+10.2f} {(sum(full_h)-sum(cos_h))/n:>+8.2f} {(sum(full_eg)-sum(cos_eg))/n:>+10.2f}")

# Qualitative notes
lines.append("")
lines.append("JUDGE NOTES:")
lines.append("-" * 60)
for jr in judge_results:
    cos_note = jr['cosine_scores'].get('note', '')
    full_note = jr['full_scores'].get('note', '')
    if cos_note or full_note:
        lines.append(f"Case {jr['case_idx']} ({jr['case_type']}):")
        if cos_note: lines.append(f"  cosine: {cos_note}")
        if full_note: lines.append(f"  full:   {full_note}")

output = "\n".join(lines)
with open('results/judge_comparison.txt', 'w') as f:
    f.write(output)

print("\n\n" + output)
print(f"\nResults saved to: results/judge_results.json + results/judge_comparison.txt")
