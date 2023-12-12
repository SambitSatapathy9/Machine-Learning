# Spam Detection Classifier

This repository contains a spam detection classifier model with Natural Language Processing (NLP) methods using Python. The tutorial primarily focuses on implementing various NLP techniques like Bag of Words (BoW), and Term Frequency-Inverse Document Frequency (TF-IDF).

## Overview

The tutorial provides step-by-step guidance and code examples on text preprocessing, feature extraction, and classification using different NLP models.

### Dependencies

- `numpy` and `pandas` for data manipulation and analysis.
- `nltk` (Natural Language Toolkit) for NLP functionalities.
- `sklearn` for machine learning models and evaluation metrics.

## Dataset Information

The dataset used in this tutorial is a tab-separated file for spam classification, comprising two columns: 'label' and 'message'. The 'message' column contains text data used for implementing NLP methods.

### Data Importing

The dataset is imported using `pandas.read_csv` with specific separator settings (`sep='\t'`) to correctly parse the tab-separated values.

```python
messages = pd.read_csv("SMSSpamCollection.txt", sep='\t', names=['label', 'message'])
```
## Steps Covered
### Text Preprocessing
1. Tokenization, Stopwords, Stemming, and Lemmatization using the NLTK library to prepare text data for analysis.
2. Application of Bag of Words (BoW), TF-IDF.
#### Regular Expression Usage
The re module is employed for pattern matching and string manipulation purposes, specifically to clean the text data by removing non-alphabetic characters.

#### Bag of Words Model Creation
A Bag of Words model is created using CountVectorizer from sklearn.feature_extraction.text. The text corpus is transformed into a numerical representation suitable for machine learning models.

#### TF-IDF Model Creation
A Term Frequency-Inverse Document Frequency (TF-IDF) model is created using TfidfVectorizer from sklearn.feature_extraction.text. This model captures the importance of words in the corpus while considering their frequency across documents.

### Machine Learning Models
#### Naive Bayes Classifier
The tutorial demonstrates the usage of the Multinomial Naive Bayes classifier for spam detection. The model is trained and evaluated using accuracy score and classification report metrics.

#### Random Forest Classifier
Additionally, the Random Forest Classifier from sklearn.ensemble is utilized to perform spam detection, and its performance is evaluated using accuracy score and classification report metrics.

### Code Execution
The code provided in this repository can be executed sequentially to understand and implement the various NLP techniques covered in the tutorial.
