import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# Đọc dữ liệu từ file CSV
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_data.csv")
df = pd.read_csv(DATA_PATH)

# 2. Tiền xử lý và chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 3. Xây dựng pipeline: TF-IDF + SVM
pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

# 4. Train mô hình
pipeline.fit(X_train, y_train)

# 5. Đánh giá
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Lưu mô hình
joblib.dump(pipeline, "sentiment_model.joblib")
