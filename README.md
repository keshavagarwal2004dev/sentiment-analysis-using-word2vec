# Sentiment Analysis: Comparative Study of Vectorization Methods

A comprehensive sentiment analysis project comparing different text vectorization approaches with multiple machine learning classifiers to identify the most effective combination for sentiment classification.

## 📋 Overview

This project performs an extensive comparative analysis of sentiment classification using:
- **4 different vectorization methods** (3 Word2Vec variants + TF-IDF)
- **4 machine learning classifiers** (Naive Bayes, Neural Network, Random Forest, KNN)
- **16 total combinations** tested to find the optimal approach

---
## 🔍 Methodology

### Data Processing Pipeline
1. **Load Dataset** - Dataset is read using pandas library
2. **Extract Reviews** - Reviews are extracted from the dataset
3. **Preprocess & Clean** - Remove HTML, digits, and punctuation
4. **Split Data** - 75% training, 25% testing (stratified split)
5. **Tokenization** - Using NLTK for word tokenization

### Vectorization Methods

#### 1. **Word2Vec (Window=3)**
- Smaller context window
- Captures local word relationships
- Better for short-range dependencies

#### 2. **Word2Vec (Window=5)**
- Medium context window (standard configuration)
- Balanced approach between local and global context
- Original baseline configuration

#### 3. **Word2Vec (Window=7)**
- Larger context window
- Captures broader contextual relationships
- Better for understanding long-range word dependencies

#### 4. **TF-IDF (Term Frequency-Inverse Document Frequency)**
- Traditional statistical approach
- Based on term importance and document rarity
- No contextual word relationship modeling

### Vector Generation Process

**For Word2Vec Models:**
- Extract embedding vector for each word in a review
- Average all word vectors to create a single review-level vector
- Results in fixed-size vector representation (128 dimensions)

**For TF-IDF:**
- Compute term frequency and inverse document frequency metrics
- Generate sparse vectors with limited features (128 for fair comparison)
- Represents document importance of each term

### Machine Learning Classifiers

1. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem with MinMax scaling
2. **Neural Network** - Multi-layer perceptron with 3 hidden layers of 10 neurons each
3. **Random Forest** - Ensemble method combining 1,000 decision trees
4. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm (k=3)

---

## 📊 Results

The project generates comprehensive performance metrics including:
- ✅ Accuracy and F1-scores for all 16 vectorization-classifier combinations
- ✅ Confusion matrices for detailed error analysis
- ✅ Precision, recall, and support metrics
- ✅ Average performance comparison by vectorization method
- ✅ Average performance comparison by classifier type
- ✅ Visual comparison charts and bar plots

Run the notebook to generate current results with your dataset!

---

## 🚀 Getting Started

### Requirements
- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, gensim, nltk, matplotlib, seaborn

### Installation
```bash
pip install pandas numpy scikit-learn gensim nltk matplotlib seaborn
```

### Usage
1. Place your dataset file in the project directory
2. Open `code.ipynb` in Jupyter Notebook
3. Run all cells to execute the complete analysis
4. Review results in the generated visualizations and metrics

---

## 📁 Project Structure

```
sentiment-analysis-using-word2vec/
├── app.py                          # Main application script
├── code.ipynb                      # Complete analysis notebook
├── README.md                       # This file
├── RESULTS_SUMMARY.md              # Summary of analysis results
├── Sentiment Analysis Dataset.tsv  # Input dataset
└── saved_models/                   # Pre-trained models
    ├── word2vec_model_w3.model     # Word2Vec (window=3)
    ├── word2vec_model_w5.model     # Word2Vec (window=5)
    └── word2vec_model_w7.model     # Word2Vec (window=7)
```

---

## 📈 Key Insights

This comparative study helps identify:
- Which vectorization method works best for sentiment analysis
- Which classifier pairs most effectively with each vectorization approach
- How context window size affects Word2Vec performance
- Trade-offs between traditional (TF-IDF) and modern (Word2Vec) approaches

---

## 🔗 References

- [Word2Vec - Mikolov et al.](https://arxiv.org/abs/1301.3781)
- [TF-IDF Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
