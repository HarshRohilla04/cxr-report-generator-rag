"""
Enrich FAISS metadata with findings and impressions from master CSV.
"""
import pickle
import pandas as pd
from pathlib import Path

def main():
    # 1. Load master CSV with text
    print("Loading mimic_master.csv...")
    df = pd.read_csv('image-data/processed/mimic_master.csv')
    print(f"  Loaded {len(df)} rows")
    
    # Create lookup by image_path
    text_lookup = {}
    for _, row in df.iterrows():
        text_lookup[row['image_path']] = {
            'findings': str(row.get('findings', '')) if pd.notna(row.get('findings')) else '',
            'impression': str(row.get('impression', '')) if pd.notna(row.get('impression')) else ''
        }
    
    # 2. Load existing metadata
    print("Loading existing FAISS metadata...")
    with open('faiss_index/mimic_faiss_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print(f"  Loaded {len(metadata)} entries")
    
    # 3. Enrich each entry
    print("Enriching metadata with text...")
    enriched_count = 0
    for entry in metadata:
        path = entry['image_path']
        if path in text_lookup:
            entry['findings'] = text_lookup[path]['findings']
            entry['impression'] = text_lookup[path]['impression']
            enriched_count += 1
    
    print(f"  Enriched {enriched_count}/{len(metadata)} entries")
    
    # 4. Backup old metadata and save new
    backup_path = 'faiss_index/mimic_faiss_metadata_backup.pkl'
    print(f"Backing up original to {backup_path}...")
    with open(backup_path, 'wb') as f:
        pickle.dump(pickle.load(open('faiss_index/mimic_faiss_metadata.pkl', 'rb')), f)
    
    print("Saving enriched metadata...")
    with open('faiss_index/mimic_faiss_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # 5. Verify
    print("\nVerification - Sample entry:")
    sample = metadata[0]
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Findings preview: {sample.get('findings', 'N/A')[:100]}...")
    print(f"  Impression preview: {sample.get('impression', 'N/A')[:100]}...")
    
    print("\n✅ Done! FAISS metadata enriched with medical text.")

if __name__ == "__main__":
    main()
