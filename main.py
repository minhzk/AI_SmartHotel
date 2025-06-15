from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmartHotel AI Service", version="1.0.0")

# Model state tracking
model_state = {
    "loaded": False,
    "loading": False,
    "error": None
}

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    original_text: str
    processed_text: Optional[str]
    sentiment_score: float
    sentiment_label: str
    confidence: Optional[float]
    model_version: Optional[str]
    processing_time_ms: Optional[int]
    keywords: Optional[List[str]]

class AnalyzeSingleRequest(BaseModel):
    review_text: str

class AnalyzeSingleResponse(BaseModel):
    review_text: str
    processed_text: Optional[str]
    predicted_rating: float
    sentiment_label: Optional[str]
    confidence: Optional[float]
    model_version: Optional[str]
    processing_time_ms: Optional[int]

async def ensure_model_loaded():
    """Đảm bảo model được load trước khi sử dụng"""
    if model_state["loaded"]:
        return True
    
    if model_state["loading"]:
        # Chờ model load xong
        import asyncio
        while model_state["loading"]:
            await asyncio.sleep(0.1)
        return model_state["loaded"]
    
    # Load model
    model_state["loading"] = True
    try:
        logger.info("🔄 Loading BERT model on demand...")
        
        # Import và khởi tạo model service
        sys.path.append(os.path.join(os.path.dirname(__file__), "service"))
        from service.bert_model_service import initialize_model
        
        # Initialize model
        initialize_model()
        
        model_state["loaded"] = True
        model_state["error"] = None
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(f"❌ {error_msg}")
        model_state["error"] = error_msg
        model_state["loaded"] = False
        
    finally:
        model_state["loading"] = False
    
    return model_state["loaded"]

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentiment(req: AnalyzeRequest):
    # Đảm bảo model đã load
    if not await ensure_model_loaded():
        raise HTTPException(
            status_code=503, 
            detail=f"Model unavailable: {model_state['error']}"
        )
    
    start = time.time()
    
    try:
        # Import function khi cần
        from service.bert_model_service import predict_review_rating_bert
        
        # Sử dụng BERT model
        result = predict_review_rating_bert(req.text)

        # Trích xuất keywords từ text (đơn giản)
        processed_text = result.get("processed_text", req.text.lower())
        keywords = processed_text.split()[:5] if processed_text else []

        processing_time_ms = int((time.time() - start) * 1000)

        return AnalyzeResponse(
            original_text=req.text,
            processed_text=processed_text,
            sentiment_score=result.get("predicted_rating"),
            sentiment_label=result.get("sentiment_label"),
            confidence=result.get("confidence"),
            model_version=result.get("model_version"),
            processing_time_ms=processing_time_ms,
            keywords=keywords,
        )
        
    except Exception as e:
        logger.error(f"❌ Error in analyze_sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-single", response_model=AnalyzeSingleResponse)
async def analyze_single_review(req: AnalyzeSingleRequest):
    # Đảm bảo model đã load
    if not await ensure_model_loaded():
        raise HTTPException(
            status_code=503, 
            detail=f"Model unavailable: {model_state['error']}"
        )
    
    start = time.time()
    
    try:
        # Import function khi cần
        from service.bert_model_service import predict_review_rating_bert
        
        # Sử dụng BERT model
        result = predict_review_rating_bert(req.review_text)
        processing_time_ms = int((time.time() - start) * 1000)
        
        return AnalyzeSingleResponse(
            review_text=req.review_text,
            processed_text=result.get("processed_text"),
            predicted_rating=result.get("predicted_rating"),
            sentiment_label=result.get("sentiment_label"),
            confidence=result.get("confidence"),
            model_version=result.get("model_version"),
            processing_time_ms=processing_time_ms,
        )
        
    except Exception as e:
        logger.error(f"❌ Error in analyze_single_review: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint - không phụ thuộc vào model"""
    return {
        "status": "healthy",
        "service": "SmartHotel AI Service",
        "version": "1.0.0",
        "timestamp": "2025-06-15",
        "port": os.environ.get("PORT", "8000")
    }

@app.get("/model-status")
async def model_status():
    """Check model loading status"""
    return {
        "model_loaded": model_state["loaded"],
        "model_loading": model_state["loading"],
        "error": model_state["error"],
        "status": "ready" if model_state["loaded"] else "not_loaded"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartHotel AI Service is running",
        "model_status": "loaded" if model_state["loaded"] else "not_loaded",
        "endpoints": ["/analyze", "/analyze-single", "/health", "/model-status"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting SmartHotel AI Service on 0.0.0.0:{port}")
    logger.info("⏳ Model sẽ được load khi có request đầu tiên")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )


"""
Để chạy chương trình và sử dụng các API xử lý AI:

1. Đảm bảo đã cài đặt các thư viện cần thiết:
   ```bash
   pip install fastapi uvicorn transformers torch scikit-learn
   ```

2. Chạy FastAPI bằng Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```
   (Chạy lệnh này trong thư mục chứa file `main.py`.)

3. Sau khi chạy, API sẽ lắng nghe tại địa chỉ:
   ```
   http://localhost:8000
   ```

4. Để kiểm tra/tương tác với API, truy cập:
   ```
   http://localhost:8000/docs
   ```
   (Swagger UI tự động sinh.)

5. Gửi request POST tới các endpoint như `/analyze-single` để phân tích bình luận.

**Lưu ý:**  
- Đảm bảo BERT model và thư mục `service` đã đúng vị trí.
- Model sẽ tự động download nếu chưa có local.
- Nếu gặp lỗi thiếu file hoặc thư viện, kiểm tra lại đường dẫn và cài đặt.
"""
