from util import *

# Add your import statements here

import nltk
from nltk.corpus import stopwords



class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		eng = stopwords.words('english')
		n = len(text)
		# print(text, 'In stopword')
		stopwordRemovedText = [[] for _ in range(n)]
		for i in range(n):
			stopwordRemovedText[i] = [word for word in text[i] if word not in eng]
		# print(text,'text')
		# stopwordRemovedText = None

		#Fill in code here

		return stopwordRemovedText




	