import numpy as np
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
import codecs
from sklearn import svm
from gensim.matutils import corpus2csc
import pickle
import pandas as pd
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

training_data_path = '../data/train.tsv'
test_data_path = '../data/test.tsv'
stop_list_path = '../data/stoplist.txt'

#==============================================================================
# Function definition
#==============================================================================
def lsi_transform(corpus, num_dims, save=True):    
    # Create a tf-idf model 
#    tfidf = models.TfidfModel(corpus) # Initialize tf-idf model
#    corpus_tfidf = tfidf[corpus] # Transform the whole corpus
#    print('TF-IDF model created.')
    
    # Create a LSI model
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_dims)
    corpus_lsi = lsi[corpus]
    print('LSI corpus created.')
    
    return lsi, corpus_lsi

def save_model(model, model_name, num_feats):
    with open('../data/' + model_name + '_model_feat' + str(num_feats) + '.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print('Model saved to: ' + '../data/' + model_name + '_model_feat' + str(num_feats) + '.pickle')

# Resampling for imbalanced dataset
# Divid the big set into num_divs sets, and upsample the small set
# Make sure in each division, the samples from big set and small set are balanced
def resampling(small_set, big_set, num_divs):
    # number of samples from each set in a division
    num_samples_div = len(big_set) // num_divs
    total_set = {}
    total_labels = {} # now the labels for small set is 1, 0 for big set
    merged_set = []
    merged_labels = []
    for i in range(num_divs):
        resampled_small = []
        for j in range(num_samples_div):
            small_idx = random.randint(0, len(small_set) - 1)
            resampled_small.append(small_set[small_idx])
        resampled_big = big_set[i*num_samples_div:(i+1)*num_samples_div]
        resampled_labels = [1] * len(resampled_small) + [0] * len(resampled_big) 
        resampled_small += resampled_big
        total_set[i] = resampled_small 
        total_labels[i] = resampled_labels
        merged_set += resampled_small
        merged_labels += resampled_labels
#    return total_set, total_labels
    return merged_set, merged_labels

#==============================================================================
# Build stop list and load data
#==============================================================================
stop_list = set()
with open(stop_list_path, "r+") as f:
    for line in f:
      if not line.strip():
    	  continue
      stop_list.add(line.strip().lower())

with codecs.open(training_data_path, "r", "utf-8") as f:
    text_list_pos = []
    text_list_neg = []
    tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')
    for line in f:
        text_not_filtered = tokenizer.tokenize(line)
        filtered_text = [word for word in text_not_filtered \
                         if word.lower() not in stop_list]
        if line[0] == '1':
            text_list_pos.append(filtered_text)
        elif line[0] == '0':
            text_list_neg.append(filtered_text)
    
    text_list = text_list_pos + text_list_neg
    
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in text_list:
        for token in text:
            frequency[token] += 1
    
    text_list = [[token for token in text if frequency[token] > 2]
                  for text in text_list]
    
    labels = [1] * len(text_list_pos) + [0] * len(text_list_neg)
    
with codecs.open(test_data_path, "r", "utf-8") as f:
    test_text_list = []

    tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')
    for line in f:
        if len(line) > 0:
            text_not_filtered = tokenizer.tokenize(line)
            filtered_text = [word for word in text_not_filtered \
                             if word.lower() not in stop_list]
            test_text_list.append(filtered_text)


    test_text_list = test_text_list[1:]

# Number of features for LSI model
num_dims = 0
#==============================================================================
# Convert text into gensim corpus and scipy sparse matrix
#==============================================================================
# Resample
#text_list, labels = resampling(text_list_pos, text_list_neg, 10)

dictionary = corpora.Dictionary(text_list)
corpus = [dictionary.doc2bow(text) for text in text_list]
corpus_sparse = corpus2csc(corpus).transpose()
# LSI transformation
#model_lsi, corpus_lsi = lsi_transform(corpus, num_dims)
#corpus_lsi_sparse = corpus2csc(corpus_lsi).transpose()

mapping = dictionary.token2id
test_corpus = [dictionary.doc2bow(text) for text in test_text_list]
test_corpus_sparse = corpus2csc(test_corpus, num_terms=len(mapping)).transpose()
# Test LSI transformation
#test_corpus_lsi = model_lsi[test_corpus]
#test_corpus_lsi_sparse = corpus2csc(test_corpus_lsi).transpose()

#==============================================================================
# Apply machine learning model
#==============================================================================
# Scaling training data for SVM training
scaler = StandardScaler()
corpus_sparse = scaler.fit_transform(corpus_sparse)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(corpus_sparse, labels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# SVM
model = svm.SVC(C=100, gamma=0.001)
model.fit(corpus_sparse, labels)

# Find out the best parameters for SVM

# Adaboost
#model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                         algorithm="SAMME",
#                         n_estimators=200)
#model.fit(corpus_sparse, labels)




training_res = model.predict(corpus_sparse)
training_acc = accuracy_score(labels, training_res)
training_rec = recall_score(labels, training_res)
training_pre = precision_score(labels, training_res)
training_f1 = f1_score(labels, training_res)

# Save model
save_model(model, 'svm', num_dims)

test_res = model.predict(test_corpus_sparse)

#s = pd.Series(test_res, index=test_text_idx, columns=['Id', 'Category'])
s = pd.DataFrame({'Category':test_res})
csv = s.to_csv('../data/test_res.csv')