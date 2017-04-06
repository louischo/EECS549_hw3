import numpy as np
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.matutils import corpus2csc
#import pickle
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

training_data_path = '../data/train.tsv'
test_data_path = '../data/test.tsv'
stop_list_path = '../data/stoplist.txt'

#==============================================================================
# Function definition
#==============================================================================
def tfidf_transform(corpus):
    # Create a tf-idf model
    tfidf = models.TfidfModel(corpus) # Initialize tf-idf model
    corpus_tfidf = tfidf[corpus] # Transform the whole corpus
    print('TF-IDF model created.')
    return corpus_tfidf
#def lsi_transform(corpus, num_dims):
#    # Create a LSI model
#    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_dims)
#    corpus_lsi = lsi[corpus]
#    print('LSI corpus created.')
#
#    return lsi, corpus_lsi
#
#def save_model(model, model_name, num_feats):
#    with open('../data/' + model_name + '_model_feat' + str(num_feats) + '.pickle', 'wb') as f:
#        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
#    print('Model saved to: ' + '../data/' + model_name + '_model_feat' + str(num_feats) + '.pickle')


#==============================================================================
# Build stop list and load data
#==============================================================================
stop_list = set()
with open(stop_list_path, "r+") as f:
    for line in f:
      if not line.strip():
    	  continue
      stop_list.add(line.strip().lower())
#stop_list = set(stopwords.words('english'))

train_data = pd.read_csv(training_data_path, sep='\t')
labels = list(train_data['label'])
comments_training = list(train_data['comment'])

tokenizer = RegexpTokenizer(r'[a-zA-Z]{1,}')
# Prepare training data
text_list = []
for line in comments_training:
    text_not_filtered = tokenizer.tokenize(line)
#    text_not_filtered = line.split()
#    filtered_text = [word.lower() for word in text_not_filtered \
#                     if word.lower() not in stop_list]
    filtered_text = [word.lower() for word in text_not_filtered]
    text_list.append(filtered_text)
#    text_list.append(text_not_filtered)

# Prepare test data
test_data = pd.read_csv(test_data_path, sep='\t')
comments_test = list(test_data['comment'])
test_text_list = []
for line in comments_test:
    text_not_filtered = tokenizer.tokenize(line)
#    text_not_filtered = line.split()
#    filtered_text = [word.lower() for word in text_not_filtered \
#                     if word.lower() not in stop_list]
    filtered_text = [word.lower() for word in text_not_filtered]
    test_text_list.append(filtered_text)
#    test_text_list.append(text_not_filtered)

# 2-gram transformation
phrases = models.phrases.Phrases(np.append(text_list, test_text_list))
bigram = models.phrases.Phraser(phrases)
phrases = models.phrases.Phrases(np.append(list(bigram[text_list]), list(bigram[test_text_list])))
trigram = models.phrases.Phraser(phrases)
text_list = list(trigram[text_list])
test_text_list = list(trigram[test_text_list])

#==============================================================================
# Find letter-level features
#==============================================================================
## Find letter-level features: training
#tokenizer = RegexpTokenizer(r'[a-zA-Z]')
#text_list_letter = []
#for line in comments_training:
#    text = tokenizer.tokenize(line)
##    text_not_filtered = line.split()
##    filtered_text = [word.lower() for word in text_not_filtered \
##                     if word.lower() not in stop_list]
#    filtered_text = [word.lower() for word in text]
#    text_list_letter.append(filtered_text)
##    text_list.append(text_not_filtered)    
#
#dictionary_letter = corpora.Dictionary(text_list_letter)
#corpus_letter = [dictionary_letter.doc2bow(text) for text in text_list_letter]
#for i in range(len(corpus_letter)):
#    corpus_letter[i] = [(corpus_letter[i][j][0], corpus_letter[i][j][1] / len(text_list_letter[i])) for j in range(len(corpus_letter[i]))]
#corpus_sparse_letter = corpus2csc(corpus_letter, num_terms=26).transpose()
#
## Find letter-level features: testing
#tokenizer = RegexpTokenizer(r'[a-zA-Z]')
#test_text_list_letter = []
#for line in comments_test:
#    text = tokenizer.tokenize(line)
##    text_not_filtered = line.split()
##    filtered_text = [word.lower() for word in text_not_filtered \
##                     if word.lower() not in stop_list]
#    filtered_text = [word.lower() for word in text]
#    test_text_list_letter.append(filtered_text)
##    text_list.append(text_not_filtered)    
#
#test_corpus_letter = [dictionary_letter.doc2bow(text) for text in test_text_list_letter]
#for i in range(len(test_corpus_letter)):
#    test_corpus_letter[i] = [(test_corpus_letter[i][j][0], test_corpus_letter[i][j][1] / len(test_text_list_letter[i])) for j in range(len(test_corpus_letter[i]))]
#test_corpus_sparse_letter = corpus2csc(test_corpus_letter, num_terms=26).transpose()
#

# Number of features for LSI model
#num_dims = 1000
#==============================================================================
# Convert text into gensim corpus and scipy sparse matrix
#==============================================================================
dictionary = corpora.Dictionary(np.append(text_list, test_text_list))
mapping = dictionary.token2id
mapping_inv = dict(zip(mapping.values(), mapping.keys()))

corpus = [dictionary.doc2bow(text) for text in text_list]
corpus = tfidf_transform(corpus)
corpus_sparse = corpus2csc(corpus, num_terms=len(mapping)).transpose()
chis, pval = chi2(corpus_sparse, labels)
pval_c = pval < 0.1
important_words = [mapping_inv[i] for i in range(len(pval_c)) if pval_c[i]]
corpus_sparse = corpus_sparse[:, pval_c]
#corpus_sparse = vstack(corpus_sparse, corpus_sparse_letter)

# LSI transformation
#model_lsi, corpus_lsi = lsi_transform(corpus, num_dims)
#corpus_lsi_sparse = corpus2csc(corpus_lsi).transpose()


test_corpus = [dictionary.doc2bow(text) for text in test_text_list]
test_corpus = tfidf_transform(test_corpus)
test_corpus_sparse = corpus2csc(test_corpus, num_terms=len(mapping)).transpose()
test_corpus_sparse = test_corpus_sparse[:, pval_c]
#test_corpus_sparse = vstack(test_corpus_sparse, test_corpus_sparse_letter)
# Test LSI transformation
#test_corpus_lsi = model_lsi[test_corpus]
#test_corpus_lsi_sparse = corpus2csc(test_corpus_lsi).transpose()

#==============================================================================
# Apply machine learning model
#==============================================================================
# Scaling training data for SVM training
#scaler = StandardScaler(with_mean=False)
#corpus_sparse = scaler.fit_transform(corpus_sparse)
#test_corpus_sparse = scaler.fit_transform(test_corpus_sparse)


#C_range = np.logspace(-2, 4, 7)
C_range = np.linspace(1,10,10)
param_grid = dict(C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(LinearSVC(penalty='l1', dual=False), param_grid=param_grid, cv=cv)
grid.fit(corpus_sparse, labels)

# with open("svm_param.txt", "w") as f:
# 	f.write("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))


# SVM
# model = SVC(C=100, gamma=0.001)

# Linear SVC
#model = LinearSVC(C=1.0, penalty='l1', dual=False, class_weight='balanced')
# Scaled
model = LinearSVC(C=8.7, penalty='l1', dual=False) 
# Not scaled pval < 0.05
model = LinearSVC(C=2.2, penalty='l1', dual=False) 
# With punctuation
model = LinearSVC(C=3.8, penalty='l1', dual=False) 
# pval < 0.1
model = LinearSVC(C=2.4, penalty='l1', dual=False) 
#model = LinearSVC(C=1.0)

# MLP
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 100, 10, 2), random_state=1)

# XGBoost sklearn
#param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
#model = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=200, gamma=0, subsample=0.8, colsample_bytree=0.8,
#					      objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
#	   	    param_grid = param_test1, n_jobs=4, iid=False, cv=5)

#model = XGBClassifier(learning_rate=0.1,
#                       n_estimators=500,
#                       max_depth=5,
#                       min_child_weight=1,
#                       gamma=0,
#                       subsample=0.8,
#                       colsample_bytree=0.8,
#                       objective= 'binary:logistic',
#                       nthread=4,
#                       scale_pos_weight=1,
#                       seed=27)

model.fit(corpus_sparse, labels)
with open("xgb_param.txt", "w") as f:
	f.write("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))


scores = cross_val_score(model, corpus_sparse, labels, cv=5)
# with open("xgb_scores.txt", "w") as f:
# 	f.write("The CV score is: %s" % (scores))

## XGBoost
## read in data
#dtrain = xgb.DMatrix(corpus_sparse, labels)
#dtest = xgb.DMatrix(test_corpus_sparse)
## specify parameters via map
#param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'binary:logistic' }
#num_round = 100
#bst = xgb.train(param, dtrain, num_round)
## make prediction
#training_res = bst.predict(dtrain)
#training_res = [int(x) for x in training_res > 0.5]




# training_res = grid.predict(corpus_sparse)
training_res = model.predict(corpus_sparse)
training_acc = accuracy_score(labels, training_res)
training_rec = recall_score(labels, training_res)
training_pre = precision_score(labels, training_res)
training_f1 = f1_score(labels, training_res)

# Save model
# save_model(model, 'svm', num_dims)
# test_res = grid.predict(test_corpus_sparse)
test_res = model.predict(test_corpus_sparse)

#s = pd.Series(test_res, index=test_text_idx, columns=['Id', 'Category'])
s = pd.DataFrame({'Category':test_res})
csv = s.to_csv('../data/test_res_xgboost.csv')
