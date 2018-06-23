# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 02:11:25 2018

@author: hp
"""

import pandas as pd    
data = pd.read_csv('spam.csv', encoding = "cp1252")
data = data[['v1', 'v2']]
data.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
count_vect = CountVectorizer()

x_train_counts = count_vect.fit_transform(data['v2'])

# Check frequesncy of a word .astype
print (count_vect.vocabulary_.get('hello'))


classifier = MultinomialNB()
classifier.fit(x_train_counts, data['v1'])

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print (classifier.predict(count_vect.transform(["Hey how are you"]))[0])


print (classifier.predict(count_vect.transform(["free credit card for you only"]))[0])


print( classifier.predict(count_vect.transform(["We are super excited to deliver happines to you. To enjoy seamless shopping experience download our app. We have a welcome gift for you. "]))[0])

print (classifier.predict(count_vect.transform(["cred!t card f0r you 0nly, great disc0unts. 0ffer on selected st0res"])))
