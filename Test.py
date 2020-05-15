import warnings
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings(action="ignore")
import os
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import en_core_web_sm

en_core_web_sm.load()
from nltk.corpus import stopwords

# PATH = 'D:\\CS580L (Int. Machine Learning)\\Project1\\aclImdb'
PATH = "C:\\Users\\chait\\OneDrive\\Documents\\Semester - 2\\ML\\Project\\Chaitanya\\aclImdb\\aclImdb"
import sys

sys.getdefaultencoding()

# Separating train files to positive and negative
posFiles = [x for x in os.listdir(PATH + "/train/pos/") if x.endswith(".txt")]
negFiles = [x for x in os.listdir(PATH + "/train/neg/") if x.endswith(".txt")]

# Separating test files to positive and negative
test_pos_Files = [x for x in os.listdir(PATH + "/test/pos/") if x.endswith(".txt")]
test_neg_Files = [x for x in os.listdir(PATH + "/test/neg/") if x.endswith(".txt")]

P_train = []
N_train = []

for nfile in negFiles:
    with open(PATH + "/train/neg/" + nfile, encoding="utf-8") as f:
        N_train.append(f.read())
for pfile in posFiles:
    with open(PATH + "/train/pos/" + pfile, encoding="utf-8") as f:
        P_train.append(f.read())

P_test = []
N_test = []
for ptestfile in test_pos_Files:
    with open(PATH + "/test/pos/" + ptestfile, encoding="utf-8") as f:
        P_test.append(f.read())
for ntestfile in test_neg_Files:
    with open(PATH + "/test/neg/" + ntestfile, encoding="utf-8") as f:
        N_test.append(f.read())

reviews_train = pd.concat([
    pd.DataFrame({"review": P_train, "Label": 1, "file": posFiles}),
    pd.DataFrame({"review": N_train, "Label": -1, "file": negFiles})
], ignore_index=True).sample(frac=1, random_state=1)

reviews_test = pd.concat([
    pd.DataFrame({"review": P_test, "Label": 1, "file": test_pos_Files}),
    pd.DataFrame({"review": N_test, "Label": -1, "file": test_neg_Files})
], ignore_index=True).sample(frac=1, random_state=1)

stopWords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()


# Data preprocessing
# Here we remove html tags, urls, special characters,Lemmanatize-
# which is better than stemmng as it gives a proper word after cutting
def rmvhtmltags(text):
    remreg = re.compile('<.*?>')
    cleartext = re.sub(remreg, '', text)
    return text


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return (vTEXT)


def rmvspclcharacter(text):
    clearspcl = re.sub(r'[^A-Za-z0-9\s.]', r'', str(text).lower())
    clearspcl = re.sub(r'\n', r' ', text)
    clearspcl = " ".join([word for word in text.split() if word not in stopWords])
    return text


def lemmatize_words(text):
    lemmatized_words = [lemmatizer.lemmatize(word, 'v') for word in text.split()]
    return ('  '.join(lemmatized_words))


# A function dataprocessing is defined where all other functions are included
def dataprocessing(x):
    x = rmvhtmltags(x)
    x = remove_urls(x)
    x = x.lower()
    x = rmvspclcharacter(x)
    x = remove_stopwords(x)
    x = strip_punctuation(x)
    x = strip_multiple_whitespaces(x)
    x = lemmatize_words(x)

    x = ' '.join([re.sub(r'\d+', '', i) for i in word_tokenize(x)])
    return x


reviews_train['review'] = reviews_train['review'].map(lambda x: dataprocessing(x))
reviews_test['review'] = reviews_test['review'].map(lambda x: dataprocessing(x))

# separating them into lists
y_train_label = reviews_train['Label'].tolist()
x_train_review = reviews_train['review'].tolist()

y_test_label = reviews_test['Label'].tolist()
x_test_review = reviews_test['review'].tolist()

X_train, X_test, y_train, y_test = train_test_split(x_train_review, y_train_label, test_size=0.3, random_state=42)

model = Pipeline([('vect', CountVectorizer()),
                  ('clf', LogisticRegression()), ])
model = model.fit(X_train, y_train)
# print("Cross Validation for Logistic regression on Count Vectorizer")
# cross_val_score(model, X_train, y_train, cv=3)

modelsvm = Pipeline([('vect', CountVectorizer()),
                     ('clf', LinearSVC()), ])
modelsvm = modelsvm.fit(X_train, y_train)
# print("Cross Validation for SVM on Count Vectorizer")
# cross_val_score(modelsvm, X_train, y_train, cv=3)

modelnb = Pipeline([('vect', CountVectorizer()),
                    ('clf', MultinomialNB()), ])
modelnb = modelnb.fit(X_train, y_train)


# print("Cross Validation for Naive Bayes on Count Vectorizer and TFID Transformer")
# cross_val_score(modelnb, X_train, y_train, cv=3)

def model_training(training_model):
    if training_model == 'LR':
        print("Training Logistic regression model using bag of words")
        lrbow = Pipeline([('vect', CountVectorizer()), ('clf', LogisticRegression()), ])
        return lrbow

    elif training_model == 'SVM':
        print("Training SVM model using bag of words")
        svmbow = Pipeline([('vect', CountVectorizer()), ('clf', LinearSVC()), ])
        return svmbow

    elif training_model == 'NB':
        print("Training NB model using bag of words")
        nbbow = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB()), ])
        return nbbow


def model_fitting_mix():
    training_model = ['LR', 'SVM', 'NB']
    for i in training_model:
        model_mix = model_training(i).fit(X_train, y_train)
        predicted_mix = model_mix.predict(X_test)
        accuracy_mix = np.mean(predicted_mix == y_test)
        scores_mix = cross_val_score(model_mix, X_train, y_train, cv=3)
        print("Accuracy on testing dataset is", accuracy_mix)
        print("Accuracy on training dataset is : %0.3f" % (scores_mix.mean()))


model_fitting_mix()

from matplotlib.pyplot import *
import matplotlib.pyplot as plt

ax = plt.subplots(1, 1, figsize=(20, 10))
accuracy_d = ['0.0', '0.852', '0.854', '0.868']
model_names = ['x', 'svm_cv', 'nb_cv', 'lr_cv']
matplotlib.pyplot.bar(model_names, accuracy_d, width=0.8, bottom=None, align='center', data=None)
plt.show()
