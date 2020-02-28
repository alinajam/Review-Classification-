import nltk
import random
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from sklearn.model_selection import KFold

from nltk.corpus import PlaintextCorpusReader
negPath = '/Users/alinajam/Desktop/DC1/neg'
posPath = '/Users/alinajam/Desktop/DC1/pos'
neglist = PlaintextCorpusReader(negPath, '.*.txt')
poslist = PlaintextCorpusReader(posPath, '.*.txt')

corpus = []
neg_corpus = [('0', list(w.lower() for w in neglist.words(fileid)))
                for fileid in neglist.fileids()]
pos_corpus = [('1', list(w.lower() for w in poslist.words(fileid)))
                for fileid in poslist.fileids()]

for sent in neg_corpus:
    corpus.append(sent)
for sent in pos_corpus:
    corpus.append(sent)

random.shuffle(corpus)

all_words = nltk.FreqDist(word.lower() for review in corpus for word in review[1])
word_features = list(all_words)#[:2000]
print(list(all_words))

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

print(document_features(corpus[0][1]))

featuresets = [(document_features(d), l) for (l, d) in corpus]
train_set, test_set = featuresets[400:], featuresets[:400]
classifier = nltk.NaiveBayesClassifier.train(train_set)


print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

#cross validation

n_splits = 10
kf = KFold(n_splits=n_splits)
cv_accuracies = []
for train, test in kf.split(featuresets):
    train_data = np.array(featuresets)[train]
    test_data = np.array(featuresets)[test]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    cv_accuracies.append(nltk.classify.accuracy(classifier, test_data))
average = sum(cv_accuracies)/n_splits

print('Accuracies with ' + str(n_splits) + '-fold cross validation: ')
for cv_accuracy in cv_accuracies:
    print(cv_accuracy)

print('Average: ' + str(average))


