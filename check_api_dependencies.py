"""
Check dependencies for SmartHotel API service
"""

import sys
import importlib
import os
from pathlib import Path


def check_library_version(library_name, import_name=None):
    """Check if library is installed and return version"""
    if import_name is None:
        import_name = library_name

    try:
        lib = importlib.import_module(import_name)
        version = getattr(lib, "__version__", "Unknown")
        print(f"✅ {library_name:<20}: {version}")
        return True, version
    except ImportError:
        print(f"❌ {library_name:<20}: Not installed")
        return False, None


def check_model_files():
    """Check if model files exist"""
    print("\n📁 Model Files Check:")

    base_dir = Path(__file__).parent
    model_files = [
        "models/bert_sentiment/config.json",
        "models/bert_sentiment/model.safetensors",
        "models/bert_sentiment/tokenizer.json",
        "models/sentiment_lstm_model.h5",
        "models/tokenizer.pkl",
        "models/label_encoder.pkl",
    ]

    for model_file in model_files:
        file_path = base_dir / model_file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_file:<40}: {size_mb:.1f} MB")
        else:
            print(f"❌ {model_file:<40}: Missing")


def main():
    print("🔍 SmartHotel API Dependencies Check")
    print("=" * 60)

    # System info
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📂 Working Directory: {os.getcwd()}")

    # FastAPI dependencies
    print("\n🚀 FastAPI Dependencies:")
    api_libs = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("python-multipart", "multipart"),
    ]

    api_ok = True
    for lib_name, import_name in api_libs:
        ok, version = check_library_version(lib_name, import_name)
        if not ok:
            api_ok = False

    # BERT model dependencies
    print("\n🤖 BERT Model Dependencies:")
    bert_libs = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("tokenizers", "tokenizers"),
    ]

    bert_ok = True
    for lib_name, import_name in bert_libs:
        ok, version = check_library_version(lib_name, import_name)
        if not ok:
            bert_ok = False

    # LSTM fallback dependencies
    print("\n🧠 LSTM Fallback Dependencies:")
    lstm_libs = [
        ("tensorflow", "tensorflow"),
        ("scikit-learn", "sklearn"),
        ("nltk", "nltk"),
    ]

    lstm_ok = True
    for lib_name, import_name in lstm_libs:
        ok, version = check_library_version(lib_name, import_name)
        if not ok:
            lstm_ok = False

    # Utility libraries
    print("\n🔧 Utility Libraries:")
    util_libs = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("requests", "requests"),
    ]

    for lib_name, import_name in util_libs:
        check_library_version(lib_name, import_name)

    # Check model files
    check_model_files()

    # Test imports
    print("\n🧪 Import Tests:")
    try:
        from service.bert_model_service import predict_review_rating_bert

        print("✅ BERT service import: OK")
    except ImportError as e:
        print(f"❌ BERT service import: {e}")

    try:
        from service.model_service import predict_review_rating

        print("✅ LSTM service import: OK")
    except ImportError as e:
        print(f"❌ LSTM service import: {e}")

    # Summary
    print("\n📊 Summary:")
    print(f"FastAPI Ready: {'✅ YES' if api_ok else '❌ NO'}")
    print(f"BERT Model Ready: {'✅ YES' if bert_ok else '❌ NO'}")
    print(f"LSTM Fallback Ready: {'✅ YES' if lstm_ok else '❌ NO'}")

    if api_ok and (bert_ok or lstm_ok):
        print("\n🎉 API service is ready to run!")
        print("Run: uvicorn main:app --reload")
    else:
        print("\n⚠️  Some dependencies are missing. Please install requirements:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
