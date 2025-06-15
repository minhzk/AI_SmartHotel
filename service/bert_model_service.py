import logging
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os

logger = logging.getLogger(__name__)

# Global variables để lưu model (lazy loading)
rating_pipeline = None
current_model_version = None
_model_initialized = False

def initialize_model():
    """Initialize model - chỉ gọi khi cần"""
    global rating_pipeline, current_model_version, _model_initialized
    
    if _model_initialized:
        logger.info("✅ Model already initialized")
        return
    
    try:
        logger.info("🔄 Initializing BERT model...")
        rating_pipeline, current_model_version = load_model()
        _model_initialized = True
        logger.info(f"✅ Model initialized successfully: {current_model_version}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise e

def load_model():
    """Load model với multiple fallbacks"""
    start_time = time.time()
    
    model = None
    tokenizer = None
    model_name = None
    
    # Danh sách models để thử (từ nhẹ đến nặng)
    model_candidates = [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert-base-uncased-finetuned-sst-2-english", 
        "nlptown/bert-base-multilingual-uncased-sentiment"
    ]
    
    for candidate in model_candidates:
        try:
            logger.info(f"🔄 Trying to load {candidate}...")
            
            # Load tokenizer trước
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            
            # Load model với config tối ưu
            model = AutoModelForSequenceClassification.from_pretrained(
                candidate,
                use_safetensors=False,  # Tắt safetensors để tránh lỗi
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                cache_dir="/tmp/transformers_cache" if os.environ.get("RAILWAY_ENVIRONMENT") else None
            )
            
            model_name = candidate
            logger.info(f"✅ Successfully loaded {candidate}")
            break
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {candidate}: {e}")
            continue
    
    if model is None or tokenizer is None:
        raise RuntimeError("Cannot load any model from the candidate list")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create pipeline
    try:
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
            framework="pt"
        )
    except Exception as e:
        logger.warning(f"⚠️ Pipeline creation failed, using manual approach: {e}")
        sentiment_pipeline = {
            "model": model,
            "tokenizer": tokenizer,
            "manual": True
        }
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
    
    version = f"{model_name.split('/')[-1]}_v1"
    return sentiment_pipeline, version

def predict_review_rating_bert(review_text: str):
    """Predict rating với model đã load"""
    global rating_pipeline, current_model_version
    
    if not _model_initialized or rating_pipeline is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")
    
    start_time = time.time()
    
    try:
        # Preprocess text
        processed_text = review_text.lower().strip()
        
        # Check if using manual approach or pipeline
        if isinstance(rating_pipeline, dict) and rating_pipeline.get("manual"):
            # Manual prediction
            result = _manual_predict(processed_text, rating_pipeline)
        else:
            # Pipeline prediction
            result = _pipeline_predict(processed_text, rating_pipeline)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format result
        return {
            "review_text": review_text,
            "processed_text": processed_text,
            "predicted_rating": result["rating"],
            "sentiment_label": result["sentiment"],
            "confidence": result["confidence"],
            "model_version": current_model_version,
            "processing_time_ms": round(processing_time, 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in predict_review_rating_bert: {e}")
        raise e

def _pipeline_predict(text: str, pipeline_model):
    """Predict using Hugging Face pipeline"""
    try:
        outputs = pipeline_model(text)
        
        # Parse output based on model type
        if isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            label = output.get('label', 'UNKNOWN')
            score = output.get('score', 0.5)
            
            # Map label to rating and sentiment
            rating, sentiment = _map_label_to_rating(label, score)
            
            return {
                "rating": rating,
                "sentiment": sentiment,
                "confidence": score
            }
        else:
            raise ValueError("Invalid pipeline output")
            
    except Exception as e:
        logger.error(f"❌ Pipeline prediction failed: {e}")
        # Fallback to neutral
        return {
            "rating": 3.0,
            "sentiment": "neutral", 
            "confidence": 0.5
        }

def _manual_predict(text: str, model_dict):
    """Manual prediction when pipeline fails"""
    try:
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get predicted class and confidence
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions).item()
        
        # Map to rating (assuming model has labels 0-4 for 1-5 stars)
        rating = float(predicted_class + 1)
        
        # Map rating to sentiment
        sentiment_map = {
            1: "very_negative",
            2: "negative", 
            3: "neutral",
            4: "positive",
            5: "very_positive"
        }
        
        sentiment = sentiment_map.get(int(rating), "neutral")
        
        return {
            "rating": rating,
            "sentiment": sentiment,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"❌ Manual prediction failed: {e}")
        # Fallback
        return {
            "rating": 3.0,
            "sentiment": "neutral",
            "confidence": 0.5
        }

def _map_label_to_rating(label: str, score: float):
    """Map model label to rating and sentiment"""
    label = label.upper()
    
    # Common label mappings
    if "POSITIVE" in label or "POS" in label:
        if score > 0.8:
            return 5.0, "very_positive"
        else:
            return 4.0, "positive"
    elif "NEGATIVE" in label or "NEG" in label:
        if score > 0.8:
            return 1.0, "very_negative"
        else:
            return 2.0, "negative"
    elif "NEUTRAL" in label:
        return 3.0, "neutral"
    else:
        # Fallback based on score
        if score > 0.7:
            return 4.0, "positive"
        elif score < 0.3:
            return 2.0, "negative"
        else:
            return 3.0, "neutral"

def get_model_status():
    """Get current model status"""
    return {
        "initialized": _model_initialized,
        "model_version": current_model_version,
        "pipeline_available": rating_pipeline is not None
    }