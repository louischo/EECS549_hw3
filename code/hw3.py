import numpy as np
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

training_data_path = '../data/train.tsv'
test_data_path = '../data/test.tsv'
stop_list_path = '../data/stoplist.txt'

#==============================================================================
# Function definition
#==============================================================================
#def tfidf_transform(corpus):
#    print('TF-IDF model created.')
#    return corpus_tfidf
#
#def build_dict(text_list):
#    dictionary = {}
#    index = 0
#    for text in text_list:
#        for word in text:
#            if word not in dictionary:
#                dictionary[word] = index
#                index += 1
#    dictionary_inv = dict(zip(dictionary.values(), dictionary.keys()))
#    return dictionary, dictionary_inv
#
#def build_corpus(dictionary, text_list):
#    corpus = np.zeros((len(text_list), len(dictionary)))
#    for i in range(len(text_list)):
#        for word in text_list[i]:
#            if word in dictionary:
#                index = dictionary[word]
#                corpus[i][index] += 1
#    return corpus
#==============================================================================
# Build stop list and load data
#==============================================================================
# Prepare training data
train_data = pd.read_csv(training_data_path, sep='\t')
labels = list(train_data['label'])
comments_training = list(train_data['comment'])

#tokenizer = RegexpTokenizer(r'[a-zA-Z]{1,}')
#text_list = []
#for line in comments_training:
#    text_not_filtered = tokenizer.tokenize(line)
#    filtered_text = [word.lower() for word in text_not_filtered]
#    text_list.append(filtered_text)


# Prepare test data
test_data = pd.read_csv(test_data_path, sep='\t')
comments_test = list(test_data['comment'])
#test_text_list = []
#for line in comments_test:
#    text_not_filtered = tokenizer.tokenize(line)
#    filtered_text = [word.lower() for word in text_not_filtered]
#    test_text_list.append(filtered_text)


#==============================================================================
# Convert text into gensim corpus and scipy sparse matrix
#==============================================================================
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,3))
corpus = vectorizer.fit_transform(comments_training)
test_corpus = vectorizer.transform(comments_test)
dictionary = vectorizer.vocabulary_
dictionary_inv = dict(zip(dictionary.values(), dictionary.keys()))
#dictionary, dictionary_inv = build_dict(text_list)
#corpus = build_corpus(dictionary, text_list)
#corpus = tfidf_transform(corpus)
#corpus_sparse = csr_matrix(corpus)
chis, pval = chi2(corpus, labels)
pval_c = pval < 0.1
important_words = [dictionary_inv[i] for i in range(len(pval_c)) if pval_c[i]]
corpus = corpus[:, pval_c]
test_corpus = test_corpus[:, pval_c]

#test_corpus = build_corpus(dictionary, test_text_list)
#test_corpus = tfidf_transform(test_corpus)
#test_corpus_sparse = test_corpus_sparse[:, pval_c]

#==============================================================================
# Apply machine learning model
#==============================================================================
# Scaling training data for SVM training
#scaler = StandardScaler(with_mean=False)
#corpus_sparse = scaler.fit_transform(corpus_sparse)
#test_corpus_sparse = scaler.fit_transform(test_corpus_sparse)


#C_range = np.logspace(-2, 4, 7)
C_range = np.linspace(4,6,21)
param_grid = dict(C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(LinearSVC(penalty='l1', dual=False), param_grid=param_grid, cv=cv)
grid.fit(corpus, labels)
print("The best parameters are %s with a score of %0.2f"
       % (grid.best_params_, grid.best_score_))

# Linear SVC
#model = LinearSVC(C=1.0, penalty='l1', dual=False, class_weight='balanced')
model = LinearSVC(C=4.0, penalty='l1', dual=False) 

model.fit(corpus, labels)

scores = cross_val_score(model, corpus, labels, cv=5)
print('scores: %s' % scores)
# training_res = grid.predict(corpus_sparse)
training_res = model.predict(corpus)
training_acc = accuracy_score(labels, training_res)
training_rec = recall_score(labels, training_res)
training_pre = precision_score(labels, training_res)
training_f1 = f1_score(labels, training_res)

test_res = model.predict(test_corpus)

s = pd.DataFrame({'Category':test_res})
csv = s.to_csv('../data/test_res_bi_0.05.csv')


