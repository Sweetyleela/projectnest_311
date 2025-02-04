import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data=pd.read_csv("/content/feedback dataset.csv")
data.head()

data.shape

data.rename(columns={'sentiment':'label'},inplace=True)

data.head()

data['label'].value_counts()

data['text'][999]

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


import nltk
nltk.download('stopwords')

stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoji_pattern.findall(text)
    text	 = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')

    prter = PorterStemmer()
    text	 = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

preprocessing('this is my tags <h1> :) <p>helo world<p> <div> <div> </h2>')


data['text'] = data['text'].apply(lambda x: preprocessing(x))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,use_idf=True,norm='l2',smooth_idf=True)
y=data.label.values
x=tfidf.fit_transform(data.text)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=cv,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)
y_pred = clf.predict(X_test)
clf.fit(X_train, y_train)

# Predicting
y_pred = clf.predict(X_test)

print(y_pred)
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import pickle
pickle.dump(clf,open('clf.pkl','wb'))
pickle.dump(tfidf,open('tfidf.pkl','wb'))

def prediction(comment):
    preprocessed_comment = preprocessing(comment)
    comment_list = [preprocessed_comment]  # Wrap the preprocessed comment in a list
    comment_vector = tfidf.transform(comment_list)
    prediction = clf.predict(comment_vector)[0]
    return prediction



prediction = prediction('hate you')

if prediction == 1:
    print("positive comment")
else:
    print("negative comment")