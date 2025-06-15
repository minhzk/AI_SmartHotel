from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import login
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
HUGGINGFACE_MODEL_NAME = "minhzk/smart_hotel"  # Replace with your repo
MODEL_VERSION = "bert_5star_v1"

# Hugging Face token (nên đặt qua biến môi trường)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Local cache path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "bert_sentiment")


def download_model_if_needed():
    """Download model from Hugging Face if not exists locally"""
    if not os.path.exists(BERT_MODEL_PATH) or not os.path.exists(os.path.join(BERT_MODEL_PATH, "config.json")):
        logger.info("Model not found locally, downloading from Hugging Face...")

        try:
            os.makedirs(BERT_MODEL_PATH, exist_ok=True)

            # Login nếu có token (cho private repo)
            if HF_TOKEN:
                login(token=HF_TOKEN)

            tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_NAME)

            tokenizer.save_pretrained(BERT_MODEL_PATH)
            model.save_pretrained(BERT_MODEL_PATH)

            logger.info("✅ Model downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    else:
        logger.info("✅ Found local model")
    return True


def load_model():
    """Load BERT model with auto-download"""
    try:
        if not download_model_if_needed():
            raise RuntimeError("Cannot download model")

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)

        rating_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            batch_size=1,
            truncation=True,
            max_length=256,
        )

        logger.info("✅ BERT model loaded successfully")
        return rating_pipeline, MODEL_VERSION

    except Exception as e:
        logger.error(f"❌ Failed to load BERT model: {e}")
        raise RuntimeError(f"Cannot initialize BERT model: {e}")


# Load model on service start
rating_pipeline, current_model_version = load_model()


def predict_review_rating_bert(review_text: str):
    """Predict star rating from review text using BERT"""
    try:
        if not review_text or not review_text.strip():
            review_text = "neutral comment"

        result = rating_pipeline(review_text)
        label = result[0]["label"]
        confidence = result[0]["score"]

        if label.startswith("LABEL_"):
            predicted_rating = int(label.split("_")[1]) + 1
        else:
            predicted_rating = 3  # fallback

        sentiment_mapping = {
            1: "very_negative",
            2: "negative",
            3: "neutral",
            4: "positive",
            5: "very_positive",
        }

        sentiment_label = sentiment_mapping.get(predicted_rating, "neutral")

        return {
            "processed_text": review_text.lower().strip(),
            "predicted_rating": float(predicted_rating),
            "sentiment_label": sentiment_label,
            "confidence": float(confidence),
            "model_version": current_model_version,
            "raw_label": label,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "processed_text": review_text.lower().strip() if review_text else "",
            "predicted_rating": 3.0,
            "sentiment_label": "neutral",
            "confidence": 0.5,
            "model_version": "error_fallback",
            "error": str(e),
        }
