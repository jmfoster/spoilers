
# coding: utf-8

# In[1]:

# Author: James Foster
# Feature Engineering for Television Spoiler Detection

from __future__ import division
import numpy as np
import pandas as pd
import classify
from IPython.core.debugger import Tracer;


# In[ ]:

# In[19]:

# Helper functions and classes

from sklearn.base import TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import SpaceTokenizer
from scipy.sparse import hstack
from collections import defaultdict

# Genre Transformer
class PageGenre(TransformerMixin):

    def __init__(self, genre_dict):
        self.genre_dict = genre_dict

    def transform(self, X, **transform_params):
        # Given page, lookup genre
        genre_lists = [self.genre_dict[page] for page in X]
        genre_strings = [" ".join(genre_list) for genre_list in genre_lists]
        return genre_strings

    def fit(self, X, y=None, **fit_params):
        return self  # does nothing

# Create polynomial features (higher order features and interactions) from
# most important features identified by select_from_model argument object.
# Optionally keep the less important features without creating polynomials for them
class AddTopPolynomials(TransformerMixin):

    def __init__(self, select_from_model, polynomial, keep_rejects=False, sparse=False):
        self.select_from_model = select_from_model
        self.polynomial = polynomial
        self.keep_rejects = keep_rejects
        self.sparse = sparse

    def transform(self, X, **transform_params):
        X_new = self.select_from_model.transform(X)
        #X_poly = self.polynomial.transform(X_new.todense())
        X_poly = self.polynomial.transform(X_new)
        if(self.keep_rejects):
            rejects_mask = [not i for i in self.select_from_model.get_support()]
            if(self.sparse):
                X_rejects = X[:, rejects_mask]
                X_poly = hstack((X_poly, X_rejects))
            else:
                X_rejects = X[:, rejects_mask]
                X_poly = np.hstack((X_poly, X_rejects))
        return X_poly

    def fit(self, X, y=None, **fit_params):
        X_new = self.select_from_model.fit_transform(X,y)
        #self.polynomial.fit(X_new.todense())
        self.polynomial.fit(X_new)
        return self

# Create Word2Vec word representations
# and compute average of words in sentence
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# Create feature that's a count of words manually selected as likely to indicate spoilers
class ManualWordsFeature(TransformerMixin):

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.words = set(["dead","die","died", 'dies','dying',"death","kill","killed","suicide","shot", "staked", "stabbed", "finale",
                          'survived', 'survive', 'stab', 'revealed', 'realize','realizes', 'reveal', 'realized', 'ends', 'crucify',
                          'averted', 'averts', 'avert', 'finds', 'finally', 'final', 'alive', 'murder', 'murdered',
                          'murderer', 'killer', 'learns', 'married', 'marries', 'wedding', 'realizes', 'actually', '!', 'pregnant',
                          'end','ending', 'killing', 'kills', 'eventually', 'reason', 'discover', 'discovered','big',
                          'averted', 'bomb', 'shoot','truth', 'ultimately', 'causes', 'affair', 'captured', 'results','fired'])

    def transform(self, X, **transform_params):
        manual_words_present = []
        for doc in X:
            doc_prepped = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
            count = len(self.words.intersection(doc_prepped))
            manual_words_present.append(count)
        return pd.DataFrame(manual_words_present)

    def fit(self, X, y=None, **fit_params):
        return self  # does nothing


# Remove words in sentence that start with capital letter (not including first word) as a
# course way to remove named entities/character names
class LowercaseLemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        keep = []
        lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        for i in range(1,len(lemmas)):
            if lemmas[i][0].islower():
                keep.append(lemmas[i])
        return keep

# Lemmatize words using WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Needed as a first step in one-hot-encoding of category strings before passing to DictVectorizer
class DictTransformer(TransformerMixin):
    """Converts passed dataframe column named in constructor to dictionary"""

    def transform(self, X, **transform_params):
        return pd.DataFrame(X).to_dict(orient='records')

    def fit(self, X, y=None, **fit_params):
        return self  # does nothing

# Extract column name(s) from passed data frame
class ColumnExtractor(TransformerMixin):
    """Takes in dataframe and returns column(s) passed in constructor"""

    def __init__(self, vars):
        self.vars=vars

    def transform(self, X):
        return X.loc[:,self.vars]

    def fit(self, X, y=None, **fit_params):
        return self

# Convert sparse matrix to dense matrix represenation
class DenseTransformer(TransformerMixin):
    """Takes in sparse matrix and returns dense matrix"""

    def transform(self, X):
        return X.todense()

    def fit(self, X, y=None, **fit_params):
        return self

# Read train and test data from file
def read_data(train, test):
    train = pd.read_csv(train, header=0)
    test = pd.read_csv(test, header=0)
    return train, test


# In[95]:
if __name__ == "__main__":

# Read in data
train, test = read_data("./data/spoilers/train.csv", "./data/spoilers/test.csv")
pages_train, pages_test = read_data("./data/spoilers/pages_train.csv", "./data/spoilers/pages_test.csv")
train = pd.concat([train, pages_train], axis=1)
test = pd.concat([test, pages_test], axis=1)


# In[16]:

import pickle
genre_dict = pickle.load(open("genre_dict.p", "rb"))




# In[96]:

# Remove tropes not in test data
shared_tropes = set(train['trope']).intersection(set(test['trope']))
shared_tropes
for i,line in test.iterrows():
    #print i, line
    #print line['trope'], test.loc[i,['trope']]
    trope = train.ix[i,'trope']
    if trope not in shared_tropes:
        train.ix[i,'trope'] = 'NotInTest'



# In[103]:

# Define Pipeline

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)

# sentenceVectorizer
sentenceVectorizer = TfidfVectorizer(strip_accents = "ascii", lowercase=True)
#sentenceVectorizer = TfidfVectorizer(lowercase=False, tokenizer=LowercaseLemmaTokenizer(), strip_accents = "ascii")
#sentenceVectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1) #bigrams
#tokenizer = LowercaseLemmaTokenizer()
#tokenizer = LemmaTokenizer()
#sentenceVectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1) #bigrams
#sentenceVectorizer = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1) #trigrams
#sentenceVectorizer = TfidfVectorizer(lowercase=False, stop_words='english', tokenizer=tokenizer)
#sentenceVectorizer = TfidfVectorizer(analyzer='char')
#sentenceVectorizer = TfidfVectorizer(stop_words='english')
#sentenceVectorizer = TfidfVectorizer()
#sentenceVectorizer = MeanEmbeddingVectorizer(w2v)
#sentenceVectorizer = CountVectorizer(stop_words='english')
#sentenceVectorizer = CountVectorizer()
#sentenceVectorizer = CountVectorizer(analyzer='char')
#sentenceVectorizer = None

# PCA
#pca = PCA(n_components=2000)
#pca = TruncatedSVD(n_components=500)
#pca = None

# Trope
tropeVectorizer = DictVectorizer(sparse=True)

# Genre
genreVectorizer = CountVectorizer()

# IMDb features
imdb_features = ['year', 'rating','votes','number of seasons']


# Select from Model: Restrict features for polynomials/interactions and for final classifier
threshold_polynomial = 6
threshold_classifier = 3
selectFromModel_polynomial = SelectFromModel(estimator=SGDClassifier(loss='log', penalty='l1', shuffle=True), threshold=threshold_polynomial)
selectFromModel_classifier = SelectFromModel(estimator=SGDClassifier(loss='log', penalty='l1', shuffle=True), threshold=threshold_classifier)

# Select K Best: Restrict features for polynomials/interactions and for final classifier
sentence_k = 'all'
trope_k = 'all'
genre_k = 'all'
classifier_k = 'all'

# Polynomial
#polynomial = None
polynomial = PolynomialFeatures(2)
#polynomial = PolynomialFeatures(2, interaction_only=True)
addTopPolynomials = AddTopPolynomials(selectFromModel_polynomial, polynomial, keep_rejects=False, sparse=False)

# Normalization
#standardizer = StandardScaler()
standardizer = MinMaxScaler()
#standardizer = None

featureUnion = FeatureUnion([
        ('sentence', Pipeline([
            ('extract', ColumnExtractor('sentence')),
            ('vectorizer', sentenceVectorizer),
#             ('to_dense', DenseTransformer()),
#             ('pca', pca),
            ('select_k', SelectKBest(k=sentence_k)),
      ])),
        ('manual_words', Pipeline([
            ('extract', ColumnExtractor('sentence')),
            ('synonym', ManualWordsFeature()),
        ])),
        ('trope', Pipeline([
            ('extract', ColumnExtractor('trope')),
            ('trope_dict', DictTransformer()),
            ('dummy', tropeVectorizer),
            ('select_k', SelectKBest(k=trope_k)),
        ])),
        ('genre', Pipeline([
            ('extract', ColumnExtractor('page')),
            ('page_lookup', PageGenre(genre_dict)),
            ('vectorizer', genreVectorizer),
            ('select_k', SelectKBest(k=genre_k)),
       ])),
       ('imdb', Pipeline([
           ('extract', ColumnExtractor(imdb_features)),
           ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
           #('poly', PolynomialFeatures(2)),
       ])),
    ])

pipeline = Pipeline([
        ('features', featureUnion),
        ('to_dense', DenseTransformer()),
        ('standardize', standardizer),
        #('add_top_polynomials', addTopPolynomials),
        #('select_from_model_classifier', selectFromModel_classifier),
        #('to_dense', DenseTransformer()), #needed for polynomial transformer or pca
        #('pca', pca),
        ('select_k', SelectKBest(k=classifier_k)),
        ('classifier', classifier)
    ])


# In[104]:

# Fit Model to Train Data
limit = .5
test_size = .2

# Split train data into train and validation data (also shuffles rows)
from sklearn.model_selection import train_test_split
train_limited = train.sample(frac=limit)
#X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(train_limited, train_limited['spoiler'], test_size=test_size)

# Fit pipeline/model to split train set
model = pipeline.fit(X_train, y_train)

# Predict spoiler for split Train and Val sets, and Compute Accuracies
from sklearn.metrics import accuracy_score
pred_train = pipeline.predict(X_train)
print "Train Accuracy:      ", round(accuracy_score(y_train, pred_train)*100,0)
pred_val = pipeline.predict(X_val)
print "Validation Accuracy: ", round(accuracy_score(y_val, pred_val)*100, 0)

# Print number of features input to final classifier
#print polynomial.n_input_features_, polynomial.n_output_features_
print "classifier feature basis: ", classifier.coef_.shape


# In[105]:

# Cross-validation
limit = 1
train_limited = train.sample(frac=limit)

from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(pipeline, train_limited, train_limited['spoiler'], cv=kf)
print("Cross-fold Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print scores


# In[79]:

# Build feature vocabulary

vocab_sentence = sentenceVectorizer.vocabulary_
#vocab_sentence = {}
vocab_manual_words = {'manual_words':0}
#vocab_manual_words = {}
vocab_trope = tropeVectorizer.vocabulary_
#vocab_trope = {}
vocab_genre = genreVectorizer.vocabulary_
#vocab_genre = {}

poly = False
mask = False
if(poly):
    vocab_inv = polynomial.get_feature_names()
else:
    # Sentence
    vocab = vocab_sentence.copy()
    # Manual Words
    for k,v in vocab_manual_words.iteritems():
        vocab[k] = v+len(vocab_sentence)
    # Trope
    for k,v in vocab_trope.iteritems():
        vocab['trope:'+k] = v+len(vocab_sentence)+len(vocab_manual_words)
    #Genre
    for k,v in vocab_genre.iteritems():
        vocab['genre:'+k] = v+len(vocab_sentence)+len(vocab_manual_words)+len(vocab_trope)
    #IMdb
    for i,f in enumerate(imdb_features):
        vocab['imdb:'+f] = i+len(vocab_sentence)+len(vocab_manual_words)+len(vocab_trope)+len(vocab_genre)
    vocab_inv = {v: k for k, v in vocab.iteritems()}

print len(vocab_sentence),'sentence'
print len(vocab_manual_words),'manual_words'
print len(vocab_trope),'trope'
print len(vocab_genre),'genre'
print len(vocab_sentence)+len(vocab_manual_words)+len(vocab_trope)+len(vocab_genre)+len(imdb_features),'sum'
print len(vocab),'vocab'
print classifier.coef_.shape[1],'final_classifier'


# In[94]:

# Print Top Features

from collections import defaultdict
from tabulate import tabulate

def analyzeFeatures(weights, n):
    """
    Return set of best features

    :param n: The number of features to return
    """
    # Best Features
    orderedFeatures = np.argsort(weights)
    bestPos = np.flipud(orderedFeatures[-n:])
    bestNeg = orderedFeatures[:n]

    # Worst Features
    orderedFeaturesAbs = np.argsort(abs(weights))
    worst = orderedFeaturesAbs[:n]

    return bestPos, bestNeg, worst


def tabulateFeatureVocabWeights(weights, features, vocab, mask=False, pretty=True):
    orig_features = features
    if(mask):
        mask_map = selectFromModel_classifier.get_support(indices=True)
        orig_features = mask_map[features]
    table = zip(features, [vocab[f] for f in orig_features if f in vocab.keys()], weights[features])

    if pretty:
        return tabulate(table, headers=['Feature #', 'Feature Name', 'Weight'])
    else:
        return weights[features]

coefs = pipeline.named_steps["classifier"].coef_.flatten()
bestPos, bestNeg, worst = analyzeFeatures(coefs,10)
print tabulateFeatureVocabWeights(coefs, bestPos, vocab_inv, mask, pretty=True),'\n'
print tabulateFeatureVocabWeights(coefs, bestNeg, vocab_inv, mask, pretty=True)
#print tabulateFeatureVocabWeights(coefs, worst[worst<len(vocab_inv)], vocab_inv)
FeatureWeights = tabulateFeatureVocabWeights(coefs, np.concatenate([bestPos,bestNeg],axis=0), vocab_inv, mask, pretty=False)
get_ipython().magic(u'Rpush FeatureWeights')


# In[81]:

get_ipython().run_cell_magic(u'R', u'', u'hist(FeatureWeights)')


# In[16]:

# Error Analysis
from sklearn.metrics import classification_report

report = classification_report(y_val, pred_val)
print(report)

# Write misclassifications to file
errors = X_train[y_train != pred_train]
errors = pd.DataFrame(errors)
errors.to_csv('./errors.csv', index=False)


# In[166]:

# Predict test set and write predictions to file
pipeline.fit(train, train['spoiler'])
pred_test = pipeline.predict(test)
predictions = np.stack((test['Id'], pred_test), axis=1)
predictions = pd.DataFrame(predictions, columns = ['Id', 'spoiler'])
predictions['spoiler'] = predictions['spoiler'].astype(bool)
predictions.to_csv('./foster_predictions.csv', index=False)


# In[ ]:

# Build genre database from IMDB
import pickle
from imdb import IMDb
import re
imdb = IMDb()

#for page in train['page']:
#    print imdb.search_movie(page)
pages = pd.concat([train['page'], test['page']])
unique_pages = set(pages)

genre_dict = defaultdict()
for page in unique_pages:
    page_title = re.sub(r'([A-Z][a-z]+)', r' \1', page).strip()
    try:
        movieID = imdb.search_movie(page_title)[0].movieID
        movie = imdb.get_movie(movieID)
        genre = movie['genre']
        title = movie['title']
        print page_title, title, genre
        genre_dict[page] = genre
    except:
        print "Exception:", page
        genre_dict[page] = [""]

pickle.dump(genre_dict, open("genre_dict.p", "wb"))


# In[ ]:

# Build movie database from IMDB
import pickle
from imdb import IMDb
import re
imdb = IMDb()

for page in unique_pages
    movieID = imdb.search_movie(page)[0].movieID
    movie = imdb.get_movie(movieID)
    print page, movie['title'], movie['genres']


# In[ ]:

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from IPython.core.debugger import Tracer
class PageIMDBVectorizer(TransformerMixin):

    def __init__(self, movie_dict_dict):
        self.movie_dict_dict = movie_dict_dict
        self.keys_to_include = ['rating','year','votes','number of seasons']

    def transform(self, X, **transform_params):
        # Given page, lookup everything about movie in imdb movie_dict
        page_vectorized = pd.DataFrame()
        for key in self.keys_to_include:
            attr_dict = {}
            for page in X:
                movie_dict = movie_dict_dict.get(page, {})
                attr_value = movie_dict.get(key, None)
                attr_dict[page] = attr_value
            dv = DictVectorizer(sparse=False)
            attr_vectorized = dv.fit_transform(attr_dict)
            attr_vectorized = pd.DataFrame(attr_vectorized)
            page_vectorized = pd.concat([page_vectorized, attr_vectorized], axis=0)
        return page_vectorized

    def fit(self, X, y=None, **fit_params):
        return self  # does nothing()


# In[ ]:

# # Convert movie dict_dict to file
# piv = PageIMDBVectorizer(movie_dict_dict)
# page_train = piv.fit_transform(train['page'])
# page_test = piv.fit_transform(test['page'])
# #pd.DataFrame(ans.todense())

# page_train.to_csv('./data/spoilers/pages_train.csv', index=False)
# page_test.to_csv('./data/spoilers/pages_test.csv', index=False)


# In[109]:

# Spell Checking
from enchant.checker import SpellChecker
chkr = SpellChecker("en_US")
for line in train['sentence']:
    chkr.set_text(line)
    for err in chkr:
        print "ERROR:", err.word


# In[31]:

get_ipython().run_cell_magic(u'R', u'', u'spoilers_by_page = tapply(train$spoiler, train$page, function(x) sum(x)/length(x))\nhist(spoilers_by_page)\nspoilers_by_page\n    \nspoilers_by_trope = tapply(train$spoiler, train$trope, function(x) sum(x)/length(x))\nhist(spoilers_by_trope)\nspoilers_by_trope')


# In[531]:

# Write most common words in spoilers to file, sorted by frequency
spoiler_sentences = train.loc[train['spoiler']==True]['sentence']
spoiler_text = ' '.join(spoiler_sentences)
spoiler_words = pd.DataFrame(word_tokenize(spoiler_text))
spoiler_word_frequencies = spoiler_words.apply(pd.value_counts)
spoiler_word_frequencies.to_csv('./spoiler_word_frequencies.csv')


# In[86]:

print ["dead","die","died", 'dies','dying',"death","kill","killed","suicide","shot", "staked", "stabbed", "finale",
                          'survived', 'survive', 'stab', 'revealed', 'realize','realizes', 'reveal', 'realized', 'ends', 'crucify',
                          'averted', 'averts', 'avert', 'finds', 'finally', 'final', 'alive', 'murder', 'murdered',
                          'murderer', 'killer', 'learns', 'married', 'marries', 'wedding', 'realizes', 'actually', '!', 'pregnant',
                          'end','ending', 'killing', 'kills', 'eventually', 'reason', 'discover', 'discovered','big',
                          'averted', 'bomb', 'shoot','truth', 'ultimately', 'causes', 'affair', 'captured', 'results','fired']


# In[ ]:
