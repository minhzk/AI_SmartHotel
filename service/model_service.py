import tensorflow as tf
import numpy as np
import pickle
import re
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Đảm bảo đường dẫn model đúng tuyệt đối (dựa trên vị trí file này)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
MAX_SEQUENCE_LENGTH = 250

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def predict_review_rating(review_text: str):
    processed_text = preprocess_text(review_text)
    seq = tokenizer.texts_to_sequences([processed_text])
    seq_pad = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    pred_prob = model.predict(seq_pad)
    pred_label = int(np.argmax(pred_prob, axis=1)[0])
    predicted_rating = float(le.inverse_transform([pred_label])[0])
    # Gán sentiment_label rõ ràng cho từng mức từ 1 đến 5
    if predicted_rating == 1:
        sentiment_label = "very_negative"
    elif predicted_rating == 2:
        sentiment_label = "negative"
    elif predicted_rating == 3:
        sentiment_label = "neutral"
    elif predicted_rating == 4:
        sentiment_label = "positive"
    elif predicted_rating == 5:
        sentiment_label = "very_positive"
    else:
        sentiment_label = "neutral"
    confidence = float(np.max(pred_prob))
    return {
        "processed_text": processed_text,
        "predicted_rating": predicted_rating,
        "sentiment_label": sentiment_label,
        "confidence": confidence,
        "model_version": "lstm_v1",
    }
