import os
from doctr.models import from_hub, ocr_predictor
from doctr.io import DocumentFile

# Force doctr to use your local cache (with the .pt files already there)
os.environ["DOCTR_CACHE_DIR"] = "weights1/hf_cache/doctr"

DET_REPO  = "Felix92/doctr-torch-db-mobilenet-v3-large"
RECO_REPO = "Felix92/doctr-torch-crnn-mobilenet-v3-large-french"

CACHE_DIR = "weights1/hf_cache"  # your snapshot_download cache

DET_REV, RECO_REV = None, None

# Load from local HuggingFace cache + doctr backbone cache
det_model  = from_hub(DET_REPO,  cache_dir=CACHE_DIR, local_files_only=True, revision=DET_REV)
reco_model = from_hub(RECO_REPO, cache_dir=CACHE_DIR, local_files_only=True, revision=RECO_REV)

predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)

doc = DocumentFile.from_images(["img test.png"])
res = predictor(doc)
print(res.render())
