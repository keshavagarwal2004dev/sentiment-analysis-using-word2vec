import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


DATA_FILE = Path("Sentiment Analysis Dataset.tsv")
MODELS_DIR = Path("saved_models")

MODEL_PATHS = {
	"Word2Vec (window=3)": MODELS_DIR / "word2vec_model_w3.model",
	"Word2Vec (window=5)": MODELS_DIR / "word2vec_model_w5.model",
	"Word2Vec (window=7)": MODELS_DIR / "word2vec_model_w7.model",
}

CLASSIFIER_PATHS = {
	"Word2Vec (window=3)": MODELS_DIR / "clf_word2vec_w3.pkl",
	"Word2Vec (window=5)": MODELS_DIR / "clf_word2vec_w5.pkl",
	"Word2Vec (window=7)": MODELS_DIR / "clf_word2vec_w7.pkl",
}

PERFORMANCE_PATH = MODELS_DIR / "classifier_performance.pkl"


def clean_text(text: str) -> str:
	text = re.sub(r"<.*?>", " ", str(text))
	text = re.sub(r"[^a-zA-Z\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip().lower()
	return text


def tokenize(text: str) -> list[str]:
	cleaned = clean_text(text)
	return cleaned.split() if cleaned else []


def average_word2vec(tokens: list[str], model: Word2Vec) -> np.ndarray:
	vectors = [model.wv[word] for word in tokens if word in model.wv]
	if not vectors:
		return np.zeros(model.vector_size, dtype=np.float32)
	return np.mean(vectors, axis=0)


@st.cache_data
def load_dataset() -> pd.DataFrame:
	if not DATA_FILE.exists():
		raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
	df = pd.read_csv(DATA_FILE, sep="\t")
	required_cols = {"sentiment", "review"}
	if not required_cols.issubset(df.columns):
		raise ValueError("Dataset must contain 'sentiment' and 'review' columns")
	df = df.dropna(subset=["review", "sentiment"]).copy()
	df["sentiment"] = df["sentiment"].astype(int)
	df["tokens"] = df["review"].apply(tokenize)
	return df


@st.cache_resource
def load_w2v_models() -> dict[str, Word2Vec]:
	models = {}
	for name, path in MODEL_PATHS.items():
		if not path.exists():
			raise FileNotFoundError(f"Model file not found: {path}")
		models[name] = Word2Vec.load(str(path))
	return models


@st.cache_resource
def load_classifiers() -> tuple[dict[str, LogisticRegression], pd.DataFrame]:
	all_classifiers_exist = all(path.exists() for path in CLASSIFIER_PATHS.values())
	performance_exists = PERFORMANCE_PATH.exists()

	if not all_classifiers_exist:
		missing = [str(path) for path in CLASSIFIER_PATHS.values() if not path.exists()]
		raise FileNotFoundError(
			"Missing saved classifier files in saved_models: " + ", ".join(missing)
		)

	if not performance_exists:
		raise FileNotFoundError(f"Missing saved performance file: {PERFORMANCE_PATH}")

	classifiers = {
		model_name: joblib.load(path)
		for model_name, path in CLASSIFIER_PATHS.items()
	}
	performance_df = joblib.load(PERFORMANCE_PATH)
	return classifiers, performance_df


def predict_sentiment(review_text: str, model_name: str) -> tuple[int, float]:
	models = load_w2v_models()
	classifiers, _ = load_classifiers()

	tokens = tokenize(review_text)
	vector = average_word2vec(tokens, models[model_name]).reshape(1, -1)
	clf = classifiers[model_name]

	sentiment = int(clf.predict(vector)[0])
	probability = float(clf.predict_proba(vector)[0][sentiment])
	return sentiment, probability


def predict_all_sentiments(review_text: str) -> pd.DataFrame:
	rows = []
	for model_name in MODEL_PATHS.keys():
		sentiment, probability = predict_sentiment(review_text, model_name)
		rows.append(
			{
				"Model": model_name,
				"Prediction": "Positive" if sentiment == 1 else "Negative",
				"Confidence": f"{probability:.2%}",
			}
		)
	return pd.DataFrame(rows)


def main() -> None:
	st.set_page_config(page_title="Sentiment Analysis (Word2Vec)", page_icon="🧠", layout="wide")

	st.title("🧠 Sentiment Analysis using Word2Vec")
	st.caption("Interactive frontend for testing sentiment predictions with saved Word2Vec embeddings.")

	try:
		df = load_dataset()
		_, performance_df = load_classifiers()
	except Exception as exc:
		st.error(f"App initialization failed: {exc}")
		st.stop()

	with st.sidebar:
		st.header("Settings")
		st.info("Predictions are shown for all 3 Word2Vec models.")
		st.markdown("---")
		st.write(f"Dataset size: **{len(df):,} reviews**")

	left_col, right_col = st.columns([2, 1])

	with left_col:
		st.subheader("Enter a Review")
		user_text = st.text_area(
			"Type/paste review text",
			height=180,
			placeholder="Example: This movie was fantastic, with great acting and a powerful story.",
		)

		if st.button("Predict Sentiment", type="primary", use_container_width=True):
			if not user_text.strip():
				st.warning("Please enter a review before predicting.")
			else:
				results_df = predict_all_sentiments(user_text)
				st.subheader("Prediction Results (All Models)")
				st.dataframe(results_df, use_container_width=True, hide_index=True)

	with right_col:
		st.subheader("Model Performance")
		st.dataframe(performance_df, use_container_width=True, hide_index=True)

	with st.expander("Preview dataset"):
		st.dataframe(df[["sentiment", "review"]].head(10), use_container_width=True)


if __name__ == "__main__":
	main()
