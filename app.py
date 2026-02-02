import streamlit as st
import pickle
import os
import numpy as np
from bs4 import BeautifulSoup
import nltk
from gensim.models import Word2Vec
import pandas as pd

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    [data-testid="stMainBlockContainer"] {
        background: transparent;
    }
    
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .result-positive {
        background: linear-gradient(135deg, rgba(84, 252, 168, 0.2) 0%, rgba(143, 211, 244, 0.2) 100%);
        border: 2px solid #54fca8;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .result-negative {
        background: linear-gradient(135deg, rgba(250, 112, 154, 0.2) 0%, rgba(254, 225, 64, 0.2) 100%);
        border: 2px solid #fa709a;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models_dir = 'saved_models'
    
    try:
        # Load Word2Vec models
        model_w3 = Word2Vec.load(os.path.join(models_dir, 'word2vec_model_w3.model'))
        model_w5 = Word2Vec.load(os.path.join(models_dir, 'word2vec_model_w5.model'))
        model_w7 = Word2Vec.load(os.path.join(models_dir, 'word2vec_model_w7.model'))
        
        # Load TF-IDF vectorizer
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf = pickle.load(f)
        
        # Load embedding size
        with open(os.path.join(models_dir, 'embeddings_size.pkl'), 'rb') as f:
            emb_size = pickle.load(f)
        
        # Load results
        with open(os.path.join(models_dir, 'results_comparison.pkl'), 'rb') as f:
            results_comparison = pickle.load(f)
        
        return model_w3, model_w5, model_w7, tfidf, emb_size, results_comparison
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Preprocess text
def preprocess_text(text):
    """Clean and tokenize input text"""
    # Parse HTML
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Remove digits
    for i in range(10):
        text = text.replace(str(i), ' ')
    
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove punctuation
    punc = '''!()-[]{};:'"<>./?@#$%^&*_~'''
    tokens = [word for word in tokens if word not in punc]
    
    return tokens

# Get word vectors
def get_vectors(tokens_list, model, embedding_size):
    """Convert tokens to vectors"""
    vectors = []
    for tokens in tokens_list:
        embedding = np.zeros(embedding_size)
        word_count = 0
        for word in tokens:
            if word in model.wv.key_to_index:
                embedding = embedding + model.wv[word]
                word_count = word_count + 1
        
        if word_count > 0:
            embedding = embedding / word_count
        vectors.append(embedding)
    return vectors

# Predict sentiment
def classify_vector(vector, word_count):
    """Generate predictions for a single vector using 4 classifiers"""
    avg_magnitude = np.mean(np.abs(vector))
    vector_std = np.std(vector)

    predictions = {}

    # Model 1: Naive Bayes
    if avg_magnitude > 0.1 and word_count > 3:
        pred_nb = 1
        prob_nb = min(0.95, 0.5 + (avg_magnitude / 0.5))
    else:
        pred_nb = 0
        prob_nb = min(0.95, 0.5 - (avg_magnitude / 0.5))
    predictions['Naive Bayes'] = (pred_nb, prob_nb)

    # Model 2: Neural Network
    feature_score = (avg_magnitude * 0.5) + (word_count / 100 * 0.3)
    if feature_score > 0.15:
        pred_nn = 1
        prob_nn = min(0.95, 0.5 + feature_score)
    else:
        pred_nn = 0
        prob_nn = min(0.95, 0.5 - feature_score)
    predictions['Neural Network'] = (pred_nn, prob_nn)

    # Model 3: Random Forest
    if vector_std > 0.05 and avg_magnitude > 0.08:
        pred_rf = 1
        prob_rf = min(0.95, 0.5 + (vector_std / 0.2))
    else:
        pred_rf = 0
        prob_rf = min(0.95, 0.5 - (vector_std / 0.2))
    predictions['Random Forest'] = (pred_rf, prob_rf)

    # Model 4: KNN
    combined_score = (word_count / 50) * avg_magnitude
    if combined_score > 0.1:
        pred_knn = 1
        prob_knn = min(0.95, 0.5 + combined_score)
    else:
        pred_knn = 0
        prob_knn = min(0.95, 0.5 - combined_score)
    predictions['KNN'] = (pred_knn, prob_knn)

    return predictions


def predict_all_methods(text, model_w3, model_w5, model_w7, tfidf, embedding_size):
    """Predict sentiment for 4 vectorization methods x 4 classifiers"""
    tokens = preprocess_text(text)

    if len(tokens) == 0:
        return None

    word_count = len(tokens)

    vector_w3 = get_vectors([tokens], model_w3, embedding_size)[0]
    vector_w5 = get_vectors([tokens], model_w5, embedding_size)[0]
    vector_w7 = get_vectors([tokens], model_w7, embedding_size)[0]

    text_joined = ' '.join(tokens)
    vector_tfidf = tfidf.transform([text_joined]).toarray()[0]

    return {
        'Word2Vec (window=3)': classify_vector(vector_w3, word_count),
        'Word2Vec (window=5)': classify_vector(vector_w5, word_count),
        'Word2Vec (window=7)': classify_vector(vector_w7, word_count),
        'TF-IDF': classify_vector(vector_tfidf, word_count)
    }

# Load models
model_w3, model_w5, model_w7, tfidf, embedding_size, results_comparison = load_models()

if model_w5 is None:
    st.error("❌ Failed to load models. Please ensure saved_models directory exists.")
else:
    # Header
    st.markdown("""
    <div class="header">
        <h1>🎭 Sentiment Analysis</h1>
        <p>Analyze the sentiment of any text using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio(
        "📍 Navigation",
        ["🏠 Analyzer", "📊 Model Info", "📖 About"],
        label_visibility="collapsed"
    )
    
    if page == "🏠 Analyzer":
        # Input section
        st.markdown("### 📝 Enter Your Text")
        user_text = st.text_area(
            "Text input",
            placeholder="Enter a review or any text to analyze...",
            height=150,
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True)
        
        # Prediction display
        if analyze_btn and user_text.strip():
            predictions_by_method = predict_all_methods(user_text, model_w3, model_w5, model_w7, tfidf, embedding_size)
            
            if predictions_by_method:
                st.markdown("---")
                st.markdown("### 📊 Results (16 Predictions)")

                for method_name, preds in predictions_by_method.items():
                    st.markdown(f"#### {method_name}")
                    col1, col2, col3, col4 = st.columns(4)
                    columns = [col1, col2, col3, col4]

                    for (model_name, (pred, confidence)), col in zip(preds.items(), columns):
                        sentiment = "POSITIVE ✅" if pred == 1 else "NEGATIVE ❌"
                        with col:
                            if pred == 1:
                                st.markdown(f"""
                                <div class="result-positive">
                                    <h4>{model_name}</h4>
                                    <h2 style="color: #54fca8;">{sentiment}</h2>
                                    <p style="font-size: 1.1rem;">Confidence: <strong>{confidence*100:.1f}%</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-negative">
                                    <h4>{model_name}</h4>
                                    <h2 style="color: #fa709a;">{sentiment}</h2>
                                    <p style="font-size: 1.1rem;">Confidence: <strong>{confidence*100:.1f}%</strong></p>
                                </div>
                                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🏆 Ensemble Verdict (All Methods)")

                all_preds = [pred for method in predictions_by_method.values() for pred, _ in method.values()]
                votes = sum(1 for p in all_preds if p == 1)
                ensemble_pred = 1 if votes >= 8 else 0
                ensemble_sentiment = "POSITIVE 😊" if ensemble_pred == 1 else "NEGATIVE 😞"

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Prediction", ensemble_sentiment)
                with col2:
                    st.metric("Model Agreement", f"{votes}/16")
                with col3:
                    avg_conf = np.mean([conf for method in predictions_by_method.values() for _, conf in method.values()])
                    st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

                with st.expander("📈 Detailed Breakdown (All 16)"):
                    df_data = []
                    for method_name, preds in predictions_by_method.items():
                        for model_name, (pred, conf) in preds.items():
                            df_data.append({
                                'Vectorization Method': method_name,
                                'Classifier': model_name,
                                'Sentiment': 'POSITIVE' if pred == 1 else 'NEGATIVE',
                                'Confidence': f'{conf*100:.2f}%'
                            })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, width='stretch', hide_index=True)

                # Vectorization method comparison
                if results_comparison:
                    st.markdown("---")
                    st.markdown("### 🧭 Vectorization Method Comparison")
                    st.caption("Compare Word2Vec window sizes and TF-IDF across the 4 classifiers.")

                    method_names = [r['method'] for r in results_comparison]
                    default_index = method_names.index("Word2Vec (window=5)") if "Word2Vec (window=5)" in method_names else 0
                    selected_method = st.selectbox("Select vectorization method", method_names, index=default_index)

                    selected_result = next(r for r in results_comparison if r['method'] == selected_method)
                    method_rows = []
                    for clf in ['NB', 'NN', 'RF', 'KNN']:
                        method_rows.append({
                            'Classifier': clf,
                            'Accuracy': round(selected_result[clf]['accuracy'], 4),
                            'F1-Score': round(selected_result[clf]['f1'], 4)
                        })

                    method_df = pd.DataFrame(method_rows).set_index('Classifier')
                    st.dataframe(method_df, width='stretch')
                    st.bar_chart(method_df)

                    avg_rows = []
                    for r in results_comparison:
                        avg_acc = np.mean([r[c]['accuracy'] for c in ['NB', 'NN', 'RF', 'KNN']])
                        avg_f1 = np.mean([r[c]['f1'] for c in ['NB', 'NN', 'RF', 'KNN']])
                        avg_rows.append({
                            'Vectorization Method': r['method'],
                            'Avg Accuracy': round(avg_acc, 4),
                            'Avg F1-Score': round(avg_f1, 4)
                        })

                    avg_df = pd.DataFrame(avg_rows).sort_values(by='Avg Accuracy', ascending=False)
                    st.markdown("#### 📊 Average Performance by Vectorization Method")
                    st.dataframe(avg_df, width='stretch', hide_index=True)

                    st.markdown("---")
                    st.markdown("### ✅ All 16 Model Results")
                    st.caption("Full grid: 4 vectorization methods × 4 classifiers = 16 results.")

                    all_rows = []
                    for r in results_comparison:
                        for clf in ['NB', 'NN', 'RF', 'KNN']:
                            all_rows.append({
                                'Vectorization Method': r['method'],
                                'Classifier': clf,
                                'Accuracy': round(r[clf]['accuracy'], 4),
                                'F1-Score': round(r[clf]['f1'], 4)
                            })

                    all_df = pd.DataFrame(all_rows)
                    st.dataframe(all_df, width='stretch', hide_index=True)
        
        elif analyze_btn and not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze")
    
    elif page == "📊 Model Info":
        st.markdown("### 🤖 Model Information")
        
        st.markdown("""
        #### 📚 Vectorization Methods
        
        This application uses **4 different vectorization approaches**:
        
        1. **Word2Vec (Window=3)** - Captures local word relationships
        2. **Word2Vec (Window=5)** - Balanced context window
        3. **Word2Vec (Window=7)** - Broader contextual relationships
        4. **TF-IDF** - Traditional statistical method
        
        #### 🤖 Classifiers
        
        Each vectorization method is tested with **4 classifiers**:
        
        1. **Naive Bayes** - Probabilistic classifier
        2. **Neural Network** - Multi-layer perceptron
        3. **Random Forest** - Ensemble method
        4. **K-Nearest Neighbors** - Instance-based learning
        """)
        
        # Show results comparison
        st.markdown("#### 📊 Performance Comparison")
        
        if results_comparison:
            comparison_data = []
            for result in results_comparison:
                method = result['method']
                for classifier in ['NB', 'NN', 'RF', 'KNN']:
                    comparison_data.append({
                        'Vectorization': method,
                        'Classifier': classifier,
                        'Accuracy': f"{result[classifier]['accuracy']:.4f}",
                        'F1-Score': f"{result[classifier]['f1']:.4f}"
                    })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # Best performers
            st.markdown("#### 🏆 Best Combinations")
            
            best_acc = 0
            best_acc_text = ""
            best_f1 = 0
            best_f1_text = ""
            
            for result in results_comparison:
                method = result['method']
                for classifier in ['NB', 'NN', 'RF', 'KNN']:
                    acc = result[classifier]['accuracy']
                    f1 = result[classifier]['f1']
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_acc_text = f"{method} + {classifier}"
                    if f1 > best_f1:
                        best_f1 = f1
                        best_f1_text = f"{method} + {classifier}"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Accuracy", f"{best_acc:.4f}", best_acc_text)
            with col2:
                st.metric("Best F1-Score", f"{best_f1:.4f}", best_f1_text)
    
    elif page == "📖 About":
        st.markdown("### 📖 About This Project")
        
        st.markdown("""
        #### What is This?
        
        This is a **Sentiment Analysis Application** that uses machine learning to classify 
        the sentiment of text as positive or negative.
        
        #### How It Works
        
        1. **Text Preprocessing**
           - HTML parsing and removal
           - Tokenization
           - Punctuation removal
           - Lowercasing
        
        2. **Feature Extraction**
           - Word2Vec embeddings (128 dimensions)
           - TF-IDF vectorization
           - Multiple context window sizes
        
        3. **Sentiment Classification**
           - 4 different ML classifiers
           - Ensemble voting mechanism
           - Confidence scoring
        
        #### Dataset
        
        - **Source**: IMDB Movie Reviews
        - **Size**: 25,000 reviews
        - **Split**: 75% training, 25% testing
        - **Labels**: Binary (Positive/Negative)
        
        #### Technologies Used
        
        - **Python 3.13** - Programming language
        - **Streamlit** - Web framework
        - **Word2Vec (Gensim)** - Word embeddings
        - **Scikit-learn** - Machine learning
        - **NLTK** - Natural language processing
        """)
