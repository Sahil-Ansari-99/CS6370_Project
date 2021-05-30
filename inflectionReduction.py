from util import *

import nltk
# Add your import statements here
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		# stemmer = PorterStemmer()
		lemmatizer = WordNetLemmatizer()
		n = len(text)
		# print(text)
		reducedText = [[] for _ in range(n)]
		for i in range(n):
			reducedText[i] = [lemmatizer.lemmatize(word) for word in text[i]]

		# reducedText = None

		#Fill in code here
		
		return reducedText


