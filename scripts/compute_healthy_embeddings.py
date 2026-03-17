import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# Load BioMedCLIP model + preprocess
model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

tokenizer = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

model = model.to(device)
model.eval()



csv_path = Path("image-data/processed/mimic_filtered_healthy_downloaded_v2.csv")
df = pd.read_csv(csv_path)

print("Healthy rows:", len(df))


def load_image(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    return img


image_root = Path("image-data/image/healthy")
batch_size = 32

all_embeddings = []
all_study_ids = []

with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]

        images = []
        for rel_path in batch_df["image_path"]:
            img_path = image_root / rel_path
            img = Image.open(img_path).convert("RGB")
            images.append(preprocess(img))

        images = torch.stack(images).to(device)

        image_embeds = model.encode_image(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        all_embeddings.append(image_embeds.cpu())
        all_study_ids.extend(batch_df["study_id"].tolist())


healthy_embeddings = torch.cat(all_embeddings, dim=0)

print("Healthy embedding matrix:", healthy_embeddings.shape)

torch.save(
    {
        "embeddings": healthy_embeddings,
        "study_ids": all_study_ids
    },
    "embeddings/mimic_healthy_embeddings.pt"
)

healthy_centroid = healthy_embeddings.mean(dim=0)
healthy_centroid = healthy_centroid / healthy_centroid.norm()

torch.save(
    healthy_centroid,
    "embeddings/mimic_healthy_centroid.pt"
)

print("Healthy centroid shape:", healthy_centroid.shape)


