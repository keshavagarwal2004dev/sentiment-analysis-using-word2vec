# Sentiment Analysis: Comparative Study of Vectorization Methods

A comprehensive sentiment analysis project comparing different text vectorization approaches with multiple machine learning classifiers.

## Overview

This project performs an extensive comparative analysis of sentiment classification using:
- **4 different vectorization methods** (3 Word2Vec variants + TF-IDF)
- **4 machine learning classifiers** (Naive Bayes, Neural Network, Random Forest, KNN)
- **16 total combinations** to find the optimal approach

## Methodology

### Data Processing Pipeline
1. Dataset is read using pandas library
2. Reviews are extracted from the dataset
3. Preprocessing and cleaning (HTML removal, digit removal, punctuation removal)
4. Data split: 75% training, 25% testing
5. Tokenization using NLTK

### Vectorization Methods

#### 1. Word2Vec (Window=3)
- Smaller context window
- Captures local word relationships
- Better for short-range dependencies

#### 2. Word2Vec (Window=5)
- Medium context window
- Balanced approach
- Original baseline configuration

#### 3. Word2Vec (Window=7)
- Larger context window
- Captures broader contextual relationships
- Better for long-range dependencies

#### 4. TF-IDF (Term Frequency-Inverse Document Frequency)
- Traditional statistical approach
- Based on term importance and rarity
- No word context consideration

### Vector Generation Process
For Word2Vec models:
- Extract embedding vector for each word in a review
- Average all word vectors to create a review-level vector
- Results in fixed-size vector representation (128 dimensions)

For TF-IDF:
- Compute term frequency and inverse document frequency
- Generate sparse vectors (limited to 128 features for fair comparison)

### Machine Learning Classifiers
1. **Naive Bayes** - Probabilistic classifier with MinMax scaling
2. **Neural Network** - Multi-layer perceptron (3 hidden layers of 10 neurons each)
3. **Random Forest** - Ensemble method with 1000 trees
4. **K-Nearest Neighbors** - Instance-based learning (k=3)

## Results

The notebook generates comprehensive comparison including:
- Detailed accuracy and F1-scores for all 16 combinations
- Confusion matrices for each classifier
- Precision, recall, and support metrics
- Average performance by vectorization method
- Average performance by classifier type
- Visual comparison charts (bar plots)

Run the notebook to see current results with your dataset!
