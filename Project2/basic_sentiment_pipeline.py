"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""

import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# used for k-fold
from sklearn.model_selection import KFold

# Hint: These are not actually used in the current
# pipeline, but would be used in an alternative
# tokenizer such as PorterStemming.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

"""
    This is a very basic tokenization strategy.

    Hint: Perhaps implement others such as PorterStemming
    Hint: Is this even used?  Where would you place it?
"""
def tokenizer(text):
    ary= []
    for word in text.split():
        if len(word) > 2:
            if word != "OED":
                ary.append(stemmer.stem(text))
    return ary

# Read in the dataset and store in a pandas dataframe
df = pd.read_csv("./training_movie_data.csv")

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.
#
# Hint: This might be an area to change the size
# of your training and test sets for improved
# predictive performance.
training_size = 35000
X_train = df.loc[:training_size, 'review'].values
y_train = df.loc[:training_size, 'sentiment'].values
X_test = df.loc[training_size:, 'review'].values
y_test = df.loc[training_size:, 'sentiment'].values

# K-Fold Cross Validation k = 10
# create training and testing set with KFold
# kf = KFold(n_splits=10)
# for train_index, test_index in kf.split(df):                # creates training sets
#     print("TRAIN:", train_index, "TEST:", test_index)       # need to figure out how to get it to work with the lr_tfidr.fit()
#     X_train, X_test = df.ix[train_index], df.ix[test_index]
#     y_train, y_test = df.ix[train_index], df.ix[test_index]


# Perform feature extraction on the text.
# Hint: Perhaps there are different preprocessors to
# test?
tfidf = TfidfVectorizer(strip_accents='unicode',
                        lowercase=False,
                        tokenizer=tokenizer,     # add tokenizer
                        stop_words = stop,       # add stop words
                        preprocessor=None)

# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression())])

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in
# sklearn or other model selection strategies.
param_grid = [{
                'vect__stop_words': [None],              # stop_words taken care of in TfidfVectorizer
                'vect__strip_accents': [None, 'ascii'],  # catch any accents missed by TfidfVectorizer
                'vect__lowercase': [False],
                'vect__preprocessor': [None],
                'vect__tokenizer': [None],               # tokenizer taken care of in TfidfVectorizer
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]
             }]

gs_lr_tfidf = GridSearchCV(lr_tfidf,
                           param_grid,
                           scoring='accuracy',
                           cv=5,                         # K-Fold k=5
                           verbose=5,                    # some messages to know it's running
                           n_jobs=-1)                    # runs computations in parallel

# Train the pipline using the training set.
gs_lr_tfidf.fit(X_train, y_train)

# Print the Test Accuracy
print('Test Accuracy: %.3f' % gs_lr_tfidf.score(X_test, y_test))

# Save the classifier for use later.
pickle.dump(gs_lr_tfidf, open("saved_model.sav", 'wb'))
