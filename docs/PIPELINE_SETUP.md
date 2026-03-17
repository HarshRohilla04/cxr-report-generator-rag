# Pipeline Setup Instructions

## Issue: Missing Dependencies

The pipeline needs `open-clip-torch` which isn't installed in your `.venv`.

## Quick Fix

Run these commands in your terminal:

```bash
cd c:\Users\Harsh\Desktop\major-project
pip install open-clip-torch
```

## Test the Fix

After installing, test if components load:

```bash
python test_pipeline_components.py
```

You should see:
```
✅ Classifier loaded successfully
✅ Abnormality detector loaded successfully  
✅ Retriever loaded successfully
✅ Embedder loaded successfully
```

## Then Run the Pipeline

Once all components load, you can use the pipeline:

```python
from pipeline_standalone import MedicalImagingPipeline

pipeline = MedicalImagingPipeline()
# Now ready to use!
```

## What Was Fixed

1. **Classifier Loader**: Updated to handle your checkpoint format which has:
   - `model_state_dict` (the actual weights)
   - `embedding_dim`, `num_classes`, `label_names` (metadata)

2. **Dependencies**: Identified missing `open-clip-torch`

## For the Notebook (05_pipeline.ipynb)

The notebook has a different issue - it tries to load embeddings incorrectly.

**Simpler solution**: Just use `pipeline_standalone.py` instead. It's more reliable.

## Next Steps

1. Install `open-clip-torch`
2. Run `test_pipeline_components.py`
3. If all tests pass, use `pipeline_standalone.py`
4. Move on to report generation experiments
