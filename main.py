from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "service"))
# Thay đổi import để sử dụng BERT model service
from service.bert_model_service import predict_review_rating_bert

app = FastAPI(title="SmartHotel AI Service", version="1.0.0")


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


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(req: AnalyzeRequest):
    start = time.time()
    # Sử dụng BERT model thay vì hardcode
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


@app.post("/analyze-single", response_model=AnalyzeSingleResponse)
def analyze_single_review(req: AnalyzeSingleRequest):
    start = time.time()
    # Sử dụng BERT model thay vì LSTM
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


@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment"""
    return {
        "status": "healthy",
        "service": "SmartHotel AI Service",
        "version": "1.0.0",
        "timestamp": "2025-05-28",
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SmartHotel AI Service is running"}


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
