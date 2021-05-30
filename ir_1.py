from util import *
import numpy as np
from collections import defaultdict
# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		# print(docs)
		N = docIDs[-1]
		# words = [word for doc in docs for sent in doc for word in sent]
		words = [word.lower() for doc in docs for sent in doc for word in sent]
		# words = [word for word in sent for sent in doc for doc in docs]
		unq_words = list(set(words))
		# print("Total Number of Unique Words : {}".format(len(unq_words)))

		wordCntDoc = [defaultdict(int) for _ in range(docIDs[-1]+1)]
		for idx in docIDs:
			for sent in docs[idx-1]:
				for word in sent:
					wordCntDoc[idx][word] += 1

		print("----------------------------------------------")
		print("Built WordCntDoc")
		print("----------------------------------------------")
		
		df = [0 for _ in range(len(unq_words)+1)]
		for i, word in enumerate(unq_words):
			cnt = 0
			for idx in docIDs:
				if wordCntDoc[idx][word] != 0:
					cnt += 1
			df[i] = cnt

		print("----------------------------------------------")
		print("Built Df")
		print("----------------------------------------------")
		
		idf = [0 for _ in range(len(unq_words)+1)]
		for i, n in enumerate(df):
			idf[i] = np.log((N+1)/(n+1))
		
		print("----------------------------------------------")
		print("Built Idf")
		print("----------------------------------------------")

		mat = np.zeros((N+1, len(unq_words)))
		for doc in docIDs:
			 for i, word in enumerate(unq_words):
				#  if wordCntDoc[doc][word] > 0:
				# 	 mat[doc][i] = idf[i]*(1+np.log(wordCntDoc[doc][word]))
				#  else:
				# 	 mat[doc][i] = 0
				#  mat[doc][i] = idf[i]*(np.log(wordCntDoc[doc][word]+1))
				 mat[doc][i] = idf[i]*(wordCntDoc[doc][word])

		print("----------------------------------------------")
		print("Built Mat")
		print("----------------------------------------------")


		self.wordCntDoc = wordCntDoc
		self.docs = docs
		self.docIDs = docIDs
		self.idf = idf
		self.index = mat
		self.unq_words = unq_words
		print("Total Vocab : ",len(unq_words))
		# self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		# Getting Vector for Query

		q = len(queries)
		doc_IDs_ordered = []
		for query in queries:
			vec = np.zeros(len(self.unq_words))
			cnt = defaultdict(int)

			for sent in query:
				for word in sent:
					# cnt[word] += 1
					cnt[word.lower()] += 1
			
			for i, word in enumerate(self.unq_words):
				vec[i] = self.idf[i]*cnt[word]
				# if word == "aeroelastic":
				# 	print(self.idf[i], cnt[word])
				# print(word, cnt[word], self.idf[i])
			scores = np.zeros(self.docIDs[-1]+1)
			scores = [[0, 0] for _ in range(self.docIDs[-1])]
			# print("Norm of query : ",np.linalg.norm(vec))
			for idx in self.docIDs:
				sc = np.dot(vec, self.index[idx,:])
				if np.linalg.norm(vec) == 0.0 or np.linalg.norm(self.index[idx,:]) == 0.0:
					scores[idx-1] = [0, idx]
					continue
				sc = sc / np.linalg.norm(vec)
				sc = sc / np.linalg.norm(self.index[idx,:])
				scores[idx-1] = [sc, idx]
				# scores[idx-1] = [sc, idx, np.linalg.norm(vec), np.linalg.norm(self.index[idx,:]),np.dot(vec, self.index[idx,:])]
			# rem = scores[485]
			scores.sort(reverse=True)
			
			# doc_IDs_ordered = []
			order = []
			for sc, idx in scores:
				order.append(idx)
			doc_IDs_ordered.append(order)
			# for idx in order[:5]:
			# 	print("------------------------")
			# 	sent = ""
			# 	for line in self.docs[idx-1]:
			# 		sent += " ".join(x for x in line)
			# 		# sent += ""
			# 	print(sent)
			
			# print(scores[:5])
			# print("Rem : ",rem)
			# print("----------------------------")
			# sent = ""
			# for line in self.docs[485]:
			# 	sent += " ".join(x for x in line)
			# 	# sent += ". "
			# print(sent)
		#Fill in code here


	
		return doc_IDs_ordered




