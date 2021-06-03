from util import *
import re
# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer


class Tokenization():
	def __init__(self):
		self.punctuations = [',', '.', '(', ')', '?', '!']

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizedText = [[] for _ in range(len(text))]
		n = len(text)
		for i in range(n):
			tokenizedText[i] = re.split('\\s+|\,|: |; |& |\.|\?|\!',text[i])
			# txt = re.split('\\s+|\,',txt)
		# tokenizedText = 

		#Fill in code here
		# print(tokenizedText)
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizer = TreebankWordTokenizer()
		tokenizedText = [[] for _ in range(len(text))]
		n = len(text)
		for i in range(n):
			tokenizedText[i] = tokenizer.tokenize(text[i])
			temp = []
			for word in tokenizedText[i]:
				if word not in self.punctuations:
					temp.append(word)
			tokenizedText[i] = temp
		#Fill in code here
		
		return tokenizedText