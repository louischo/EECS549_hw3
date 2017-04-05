import numpy as np
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

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
def lsi_transform(corpus, num_dims):    
    # Create a LSI model
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_dims)
    corpus_lsi = lsi[corpus]
    print('LSI corpus created.')
    
    return lsi, corpus_lsi
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

train_data = pd.read_csv(training_data_path, sep='\t')
labels = list(train_data['label'])
comments = list(train_data['comment'])

tokenizer = RegexpTokenizer(r'[a-zA-Z]{1,}')
# Prepare training data
text_list = []
for line in comments:
    text_not_filtered = tokenizer.tokenize(line)
#    filtered_text = [word.lower() for word in text_not_filtered \
#                     if word.lower() not in stop_list]
    filtered_text = [word.lower() for word in text_not_filtered]
    text_list.append(filtered_text)
#    text_list.append(text_not_filtered)

test_data = pd.read_csv(test_data_path, sep='\t')
comments = list(test_data['comment'])
test_text_list = []
for line in comments:
    text_not_filtered = tokenizer.tokenize(line)
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

# Number of features for LSI model
#num_dims = 1000
#==============================================================================
# Convert text into gensim corpus and scipy sparse matrix
#==============================================================================
dictionary = corpora.Dictionary(np.append(text_list, test_text_list))
mapping = dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in text_list]
corpus = tfidf_transform(corpus)
corpus_sparse = corpus2csc(corpus, num_terms=len(mapping)).transpose()
# LSI transformation
#model_lsi, corpus_lsi = lsi_transform(corpus, num_dims)
#corpus_lsi_sparse = corpus2csc(corpus_lsi).transpose()


test_corpus = [dictionary.doc2bow(text) for text in test_text_list]
test_corpus = tfidf_transform(test_corpus)
test_corpus_sparse = corpus2csc(test_corpus, num_terms=len(mapping)).transpose()
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
C_range = np.linspace(0.7,0.8,11)
param_grid = dict(C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(LinearSVC(penalty='l1', dual=False), param_grid=param_grid, cv=cv)
grid.fit(corpus_sparse, labels)

with open("svm_param.txt", "w") as f:
	f.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# SVM
# model = SVC(C=100, gamma=0.001)

# Linear SVC
#model = LinearSVC(C=1.0, penalty='l1', dual=False, class_weight='balanced')
model = LinearSVC(C=0.78, penalty='l1', dual=False)
#model = LinearSVC(C=1.0)

# MLP
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 100, 10, 2), random_state=1)

# Adaboost
#model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
model.fit(corpus_sparse, labels)
scores = cross_val_score(model, corpus_sparse, labels, cv=5)

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



training_res = grid.predict(corpus_sparse)
training_res = model.predict(corpus_sparse)
training_acc = accuracy_score(labels, training_res)
training_rec = recall_score(labels, training_res)
training_pre = precision_score(labels, training_res)
training_f1 = f1_score(labels, training_res)

# Save model
# save_model(model, 'svm', num_dims)
test_res = grid.predict(test_corpus_sparse)
test_res = model.predict(test_corpus_sparse)

#s = pd.Series(test_res, index=test_text_idx, columns=['Id', 'Category'])
s = pd.DataFrame({'Category':test_res})
csv = s.to_csv('../data/test_res_svc_c1.csv')
