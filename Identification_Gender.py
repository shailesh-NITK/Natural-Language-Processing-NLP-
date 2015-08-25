"""
NLP is used for identification of genders.
Inputs are male and female name.

    @shailesh-NITK

"""


import nltk
import random
from show import show

def g_f(word):
    return {'last_letter': word[-1]}

names = ([(name, 'male') for name in nltk.corpus.names.words('male.txt')] + \
        [(name, 'female') for name in nltk.corpus.names.words('female.txt')])
random.shuffle(names)
show(names[0:4])

feature = [(g_f(n), g) for (n,g) in names]
train_set= feature[:500]
test_set = feature[500:],

classifier = nltk.NaiveBayesClassifier.train(train_set)

print classifier.classify(g_f('Neo'))
# 'is_male'
print classifier.classify(g_f('Trinity'))
# 'is_female'
print nltk.classify.accuracy(classifier, test_set)
# 0.758
classifier.show_most_informative_features(5)

# Most Informative Features
#              last_letter = 'a'            female : male   =     38.3 : 1.0
#              last_letter = 'k'              male : female =     31.4 : 1.0
#              last_letter = 'f'              male : female =     15.3 : 1.0
#              last_letter = 'p'              male : female =     10.6 : 1.0
#              last_letter = 'w'              male : female =     10.6 : 1.0



from nltk.classify import apply_features
train_set = apply_features(g_f, names[500:])
test_set = apply_features(g_f, names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)

print classifier.classify(g_f('Neo'))
# 'is_male'
print classifier.classify(g_f('Trinity'))
# 'is_female'
print nltk.classify.accuracy(classifier, test_set)
# 0.758
classifier.show_most_informative_features(5)
# Most Informative Features
#              last_letter = 'a'            female : male   =     38.3 : 1.0
#              last_letter = 'k'              male : female =     31.4 : 1.0
#              last_letter = 'f'              male : female =     15.3 : 1.0
#              last_letter = 'p'              male : female =     10.6 : 1.0
#              last_letter = 'w'              male : female =     10.6 : 1.0

def g_f2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features

print str(g_f2('shailesh'))[0:100]
# {'count(j)': 1, 'has(d)': False, 'count(b)': 0, ...}

"""
conclusion:
Feature sets returned by g_f2 contain a large
number of specific features, so it will overfit the small
Names corpus.
Accuracy of the classifier on the test set using g_f2
is lower than when gender_features was used,
0.748 < 0.758.
"""

random.shuffle(names)
featuresets = [(g_f2(n), g) for (n,g) in names]
train_set = feature[:500]
test_set = feature[500:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
# 0.748

"""
Can use error analysis to improve classifier.

Divide up into
- training set
- dev-test / cross validation set
- test set
"""

random.shuffle(names)
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]

train_set = [(g_f(n), g) for (n,g) in train_names]
devtest_set = [(g_f(n), g) for (n,g) in devtest_names]
test_set = [(g_f(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set) 
print nltk.classify.accuracy(classifier, devtest_set) 
# 0.765

"""
Generate list of errors.
"""

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(g_f(name))
    if guess != tag:
        errors.append( (tag, guess, name) )

for (tag, guess, name) in sorted(errors)[0:5]: 
    print 'correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name)
# correct=female   guess=male     name=Cindelyn
# correct=female   guess=male     name=Katheryn
# correct=female   guess=male     name=Kathryn
# correct=male     guess=female   name=Aldrich
# correct=male     guess=female   name=Mitch
# correct=male     guess=female   name=Rich

"""
Errors show that:

eg. -n tends to be male bu -yn tend to be female
Hence error analysis indicates two character features are
important
"""

def g_f1(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}

"""
Testing shows the accuracy is improved by including two
character suffix features.
Use a different dev-test/training split to avoid overfitting.
"""

random.shuffle(names)
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]

train_set = [(g_f1(n), g) for (n,g) in train_names]
devtest_set = [(g_f1(n), g) for (n,g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, devtest_set)
# 0.782
