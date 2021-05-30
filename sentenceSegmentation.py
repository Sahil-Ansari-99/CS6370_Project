from util import *
import nltk
import re
# Add your import statements here




class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		segmentedText = re.split('\!|\?|\.|\n',text)
		#Fill in code here

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
		segmentedText = tokenizer.tokenize(text)
		#Fill in code here
		
		return segmentedText