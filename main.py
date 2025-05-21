from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time

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
