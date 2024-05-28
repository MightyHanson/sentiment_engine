# Required Libraries
# web scraper package
from typing import Optional
import requests
from bs4 import BeautifulSoup
import re
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
# email filter package
import imaplib
from email import policy
from email.parser import BytesParser
from datetime import datetime, timedelta
import email.utils
# pdf
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
# we need some NLP for the pdf
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