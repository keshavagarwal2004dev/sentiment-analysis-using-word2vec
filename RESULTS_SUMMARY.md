# Results Summary - Sentiment Analysis Comparative Study

## Experiment Overview

**Date**: Project Updated for Comprehensive Comparison
**Dataset**: Sentiment Analysis Dataset (TSV format)
**Train/Test Split**: 75% / 25%

## Vectorization Methods Tested

| Method | Description | Vector Size | Context Window |
|--------|-------------|-------------|----------------|
| Word2Vec (w=3) | Small context window | 128 | 3 words |
| Word2Vec (w=5) | Medium context window | 128 | 5 words |
| Word2Vec (w=7) | Large context window | 128 | 7 words |
| TF-IDF | Statistical frequency-based | 128 | N/A |

## Classifiers Tested

1. **Naive Bayes (NB)** - Multinomial with MinMax scaling
2. **Neural Network (NN)** - MLP with 3 hidden layers (10-10-10)
3. **Random Forest (RF)** - 1000 trees
4. **K-Nearest Neighbors (KNN)** - k=3

## Total Experiments: 16 Combinations

(4 vectorization methods × 4 classifiers)

## Key Findings

### Run the notebook to generate results!

The notebook will automatically:
- Train all 16 model combinations
- Display detailed metrics for each:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - Macro and Micro F1
- Generate comparison tables
- Identify best performers
- Create visualization charts
- Save all models for future use

## Saved Models

All trained models are saved in `saved_models/` directory:
- `word2vec_model_w3.model` - Word2Vec with window=3
- `word2vec_model_w5.model` - Word2Vec with window=5
- `word2vec_model_w7.model` - Word2Vec with window=7
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `results_comparison.pkl` - All results data
- `embeddings_size.pkl` - Embedding dimension configuration

## How to Use

1. Open `code.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. View comprehensive results and comparisons
4. Check `results_comparison.png` for visual analysis

## Insights to Look For

- Which context window size works best for Word2Vec?
- How does Word2Vec compare to traditional TF-IDF?
- Which classifier performs best across all methods?
- Are there method-classifier combinations that work particularly well?
- Does increasing context window always improve performance?

## Next Steps

After running the notebook, you can:
- Experiment with different hyperparameters
- Try additional vectorization methods (Doc2Vec, BERT embeddings, etc.)
- Test other classifiers (SVM, XGBoost, etc.)
- Perform cross-validation for more robust results
- Analyze misclassified examples
