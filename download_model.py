# download_model.py
from huggingface_hub import snapshot_download

# Pick a smaller model so Render can handle it quickly
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # ~300MB

# Download and store locally in ./models/
snapshot_download(MODEL_NAME, local_dir="./models/distilbart")
print(f"✅ Model {MODEL_NAME} downloaded to ./models/distilbart")
