from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "service"))
from service.model_service import predict_review_rating

app = FastAPI()


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
    # TODO: Thay thế bằng mô hình thực tế của bạn
    processed_text = req.text.lower()
    sentiment_score = 7.5  # ví dụ
    sentiment_label = "positive"
    confidence = 0.95
    model_version = "1.0.0"
    keywords = ["hotel", "service"]
    processing_time_ms = int((time.time() - start) * 1000)
    return AnalyzeResponse(
        original_text=req.text,
        processed_text=processed_text,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        confidence=confidence,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
        keywords=keywords,
    )


@app.post("/analyze-single", response_model=AnalyzeSingleResponse)
def analyze_single_review(req: AnalyzeSingleRequest):
    start = time.time()
    # Gọi service xử lý model thực tế
    result = predict_review_rating(req.review_text)
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


"""
Để chạy chương trình và sử dụng các API xử lý AI:

1. Đảm bảo đã cài đặt các thư viện cần thiết:
   ```bash
   pip install fastapi uvicorn tensorflow scikit-learn nltk
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
- Đảm bảo các file model, tokenizer, label_encoder và thư mục `service` đã đúng vị trí.
- Nếu gặp lỗi thiếu file hoặc thư viện, kiểm tra lại đường dẫn và cài đặt.
"""
