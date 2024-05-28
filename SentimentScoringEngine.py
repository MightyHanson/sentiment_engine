from TextObject import TextObject
# we need some NLP for the pdf
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
# Download the Reuters dataset
from nltk.corpus import reuters
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('reuters')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon')
from typing import Optional
import string
from sklearn.feature_extraction.text import TfidfVectorizer
# word2vec
from gensim.models import Word2Vec
import numpy as np
# model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
# one new source
import pysentiment2 as ps
# add BERT
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Build Sentiment Scoring Engine: determine positive/negative/netural from the given article
# In further development, it is designed to find out which asset this sentiment impacts
# According to the requirement
# Tokenization: Convert the text into tokens (words).
# Word Embeddings: Convert tokens into vectors using embeddings (e.g., Word2Vec).
# Sentiment Scoring: Use a pre-trained sentiment analysis model to get the sentiment score.

class SentimentScoringEngine:
    def __init__(self, model_path=None):
        self.lemmatizer = WordNetLemmatizer()
        # word2vec
        self.model_path = model_path
        if model_path:
            self.w2v_model = Word2Vec.load(model_path)
        else:
            self.w2v_model = None
            # Load opinion lexicon
        self.positive_words = set(opinion_lexicon.positive())
        self.negative_words = set(opinion_lexicon.negative())
        # adding Harvard IV-4 dictionary & Loughran and McDonald dictionary
        self.lm = ps.LM()
        self.hiv4 = ps.HIV4()
        # BERT initialization
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased').eval()
        # Replace the BERT model (to be faster) and tokenizer with DistilBERT's
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').eval()

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        Convert treebank POS tagging to first character used by WordNetLemmatizer
        """
        switcher = {
            'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.ADV,
        }
        return switcher.get(treebank_tag[0], wordnet.NOUN)

    def preprocess_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text)
        # Convert tokens to lowercase
        tokens = [word.lower() for word in tokens]
        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        # according to google search, For sentiment analysis, lemmatization is often preferred
        # because it produces more accurate root words.
        # Lemmatization
        lemmatized_tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) for word, pos in pos_tags]
        return lemmatized_tokens
    # Feature Extraction, use w2v model because we have large dataset, compared with the pretrained model
    def get_word_vector(self, word):
        try:
            return self.w2v_model.wv[word]
        except KeyError:
            return np.zeros(self.w2v_model.vector_size)

    # 3-dimension [batch_size, sequence_length, embedding_size] to 2 [batch_size, embedding_size]
    # according to Logistic Regression requires [number_of_samples, number_of_features]
    def get_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        # Average over the sequence length
        embeddings = outputs['last_hidden_state'].mean(dim=1)
        # Convert tensor to numpy and flatten it
        return embeddings.numpy().squeeze()

    def get_document_vector(self, text, use_bert=False):
        if use_bert:
            return self.get_bert_embedding(text)
        else:
            words = self.preprocess_text(text)
            vectors = [self.get_word_vector(word) for word in words]
            return np.mean(vectors, axis=0)

    def get_lm_score(self, article):
        tokens = self.lm.tokenize(article)
        score = self.lm.get_score(tokens)
        return score

    def get_hiv4_score(self, article):
        tokens = self.hiv4.tokenize(article)
        score = self.hiv4.get_score(tokens)
        return score
    def assign_label(self, article):
        def score_comparsion(pos_count,neg_count):
            if pos_count>neg_count:
                return 1
            elif pos_count<neg_count:
                return -1
            else:
                return 0
        words = self.preprocess_text(article)
        positive_count = sum(word in self.positive_words for word in words)
        negative_count = sum(word in self.negative_words for word in words)
        # Get scores from Loughran-McDonald and Harvard IV-4 dictionaries
        lm_score = self.get_lm_score(article)
        hiv4_score = self.get_hiv4_score(article)

        # Combine scores
        # combined_score = positive_count + lm_score['Positive'] + hiv4_score['Positive'] - negative_count - lm_score[
        #     'Negative'] - hiv4_score['Negative']
        # I design a voting mechanism, from three dictionaries only 2/3 can conclude the direction
        combined_score = score_comparsion(positive_count,negative_count) + score_comparsion(
            lm_score['Positive'],lm_score['Negative']) + score_comparsion(hiv4_score['Positive'],hiv4_score['Negative'])

        if combined_score > 0:
            return "Positive"
        elif combined_score < 0:
            return "Negative"
        else:
            return "Neutral"


    def initialize_and_train_w2v(self, input_comb_text):
        if os.path.exists(self.model_path):
            self.w2v_model = Word2Vec.load(self.model_path)
            self.w2v_model.build_vocab(input_comb_text, update=True)
            self.w2v_model.train(input_comb_text, total_examples=self.w2v_model.corpus_count, epochs=10)
        else:
            self.w2v_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
            self.w2v_model.build_vocab(input_comb_text)
            self.w2v_model.train(input_comb_text, total_examples=self.w2v_model.corpus_count, epochs=10)
            if not os.path.exists("w2v_model"):
                os.makedirs("w2v_model")
        self.w2v_model.save(self.model_path)

    def extract_features_and_labels(self, input_comb_text, use_bert):
        X = [self.get_document_vector(article,use_bert) for article in input_comb_text]
        y = [self.assign_label(article) for article in input_comb_text]
        return X, y

    def train_and_evaluate_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=16)

        clf_logit = LogisticRegression(max_iter=1000)
        clf_logit.fit(X_train, y_train)
        y_pred_logit = clf_logit.predict(X_test)
        accuracy_logit = accuracy_score(y_test, y_pred_logit)
        print(f"Accuracy from Logit regression: {accuracy_logit * 100:.2f}%")
        print(f"F1 Score for Logit regression: {f1_score(y_test, y_pred_logit, average='weighted'):.2f}")
        print(
            f"ROC AUC for Logit regression: {roc_auc_score(y_test, clf_logit.predict_proba(X_test), multi_class='ovr'):.2f}")
        print("Classification Report for Logit regression:")
        print(classification_report(y_test, y_pred_logit))
        print("Confusion Matrix for Logit regression:")
        print(confusion_matrix(y_test, y_pred_logit))

        clf_rf = RandomForestClassifier(class_weight='balanced', random_state=16)
        clf_rf.fit(X_train, y_train)
        y_pred_rf = clf_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        print(f"Accuracy from random forest classifier: {accuracy_rf * 100:.2f}%")
        print(f"F1 Score for Random Forest: {f1_score(y_test, y_pred_rf, average='weighted'):.2f}")
        print(
            f"ROC AUC for Random Forest: {roc_auc_score(y_test, clf_rf.predict_proba(X_test), multi_class='ovr'):.2f}")
        print("Classification Report for Random Forest:")
        print(classification_report(y_test, y_pred_rf))
        print("Confusion Matrix for Random Forest:")
        print(confusion_matrix(y_test, y_pred_rf))

        return clf_logit, clf_rf

    def train_and_evaluate(self, input_comb_text, use_bert=False):
        if not use_bert:
            self.initialize_and_train_w2v(input_comb_text)
        X, y = self.extract_features_and_labels(input_comb_text, use_bert=use_bert)
        label_counts = Counter(y)
        print(label_counts)
        return self.train_and_evaluate_models(X, y)


