from util import *
import numpy as np
from collections import defaultdict
from collections import Counter
from nltk.corpus import wordnet
from sklearn.cluster import KMeans
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
		# print(docIDs)
		m = len(docIDs)
		for i in range(m):
			docIDs[i] -= 1
		# print(docIDs)
		N = docIDs[-1]
		# words = [word for doc in docs for sent in doc for word in sent]
		words = []
		for doc in docs:
			for sent in doc:
				for word in sent:
					if not word.isalpha():
						if '-' in word:
							w1, w2 = word.split('-')[0], word.split('-')[1]
							words.append(w1.lower())
							words.append(w2.lower())
					else:
						words.append(word.lower())
		# words = [word for doc in docs for sent in doc for word in sent if word != "." and word !=","]
		# notwords = [word for doc in docs for sent in doc for word in sent if not word.isalpha()]
		words = [wordnet.synsets(word.lower())[0].name() if len(wordnet.synsets(word.lower())) > 0 else word.lower() for word in words]
		# words = [word.lower() for word in words]
		# words = [word.lower() for doc in docs for sent in doc for word in sent]
		# bigrams = [sent[i].lower()+" "+sent[i+1].lower() for doc in docs for sent in doc for i in range(len(sent)-1)]
		# unq_bigrams = list(set(bigrams))
		# print("Total No of Bigrams : ",len(unq_bigrams))
		# words = [word for word in sent for sent in doc for doc in docs]
		unq_words = list(set(words))
		# unq_words = [word for word in unq_words if word != "." or word != ","]
		# words = [word for word in words if word.isalpha()]
		# words = [word for word in words if word != "." or word != ","]
		ct = Counter(words)
		# print(ct.most_common()[:200])
		# print("Total Number of Unique Words : {}".format(len(unq_words)))

		wordCntDoc = [defaultdict(int) for _ in range(docIDs[-1]+1)]
		# pij = [defaultdict(int) for _ in range(docIDs[-1]+1)]
		for idx in docIDs:
			for sent in docs[idx]:
				for word in sent:
					# if len(wordnet.synsets(word.lower())) > 0:
					# 	wordCntDoc[idx][wordnet.synsets(word.lower())[0].name()] += 1
					# else:
					# 	wordCntDoc[idx][word.lower()] += 1
					# wordCntDoc[idx][word.lower()] += 1
					if word.isalpha():
						if len(wordnet.synsets(word.lower())) > 0:
							wordCntDoc[idx][wordnet.synsets(word.lower())[0].name()] += 1
						else:
							wordCntDoc[idx][word.lower()] += 1
					else:
						sword = word.lower()
						if '-' in sword:
							l = sword.split("-")
							w1, w2 = l[0], l[1]
							if len(wordnet.synsets(w1.lower())) > 0:
								wordCntDoc[idx][wordnet.synsets(w1.lower())[0].name()] += 1
							else:
								wordCntDoc[idx][w1.lower()] += 1
							if len(wordnet.synsets(w2.lower())) > 0:
								wordCntDoc[idx][wordnet.synsets(w2.lower())[0].name()] += 1
							else:
								wordCntDoc[idx][w2.lower()] += 1

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

		gf = [0 for _ in range(len(unq_words)+1)]
		for i, word in enumerate(unq_words):
			cnt = 0
			for idx in docIDs:
				cnt += wordCntDoc[idx][word]
			if cnt == 0:
				print("word Zero:",word)
			gf[i] = cnt
		# print(gf[:100])
		print("----------------------------------------------")
		print("Built Df")
		print("----------------------------------------------")
		
		idf = [0 for _ in range(len(unq_words)+1)]
		for i, n in enumerate(df):
			idf[i] = np.log((N+1)/(n+1))
		
		print("----------------------------------------------")
		print("Built Idf")
		print("----------------------------------------------")

		entropy = [0 for _ in range(len(unq_words)+1)]
		for i, word in enumerate(unq_words):
			res = 1
			for idx in docIDs:
				pij = wordCntDoc[idx][word]/gf[i]
				if np.log(N) == 0:
					print("N")
				res += (np.log(pij+1)*pij)/np.log(N)
			
			entropy[i] = res
		#Term Document Matrix Mat
		# mat = np.zeros((N+1, len(unq_words)))
		mat = np.zeros((len(unq_words), N+1))
		for doc in docIDs:
			for i, word in enumerate(unq_words):
				if wordCntDoc[doc][word] > 0:
					mat[i][doc] = entropy[i]*(1 + np.log(wordCntDoc[doc][word]))
				else:
					mat[i][doc] = 0
				#  mat[i][doc] = idf[i]*(np.log(wordCntDoc[doc][word]+1))
				#  mat[i][doc] = idf[i]*(wordCntDoc[doc][word])

		print("----------------------------------------------")
		print("Built Mat")
		print("----------------------------------------------")

		U, s, Vh = np.linalg.svd(mat)

		# K Components
		print("Mat shape:",mat.shape,"U shape:",U.shape)
		print("S shape:",np.diag(s).shape, "Vh shape:",Vh.shape)
		self.k = 300

		# matd = U @ np.diag(s) @ Vh
		print("Check",np.allclose(mat, np.dot(U[:, :s.shape[0]] * s, Vh)))

		# print("Difference :",np.linalg.norm(matd-mat))

		self.s = np.zeros((U.shape[0], Vh.shape[0]))

		self.sigma = np.zeros((self.k,self.k))

		for i in range(self.k):
			self.s[i][i] = s[i]
			self.sigma[i][i] = s[i]

		self.U = U @ self.s
		self.U = self.U[:,:self.k]
		self.Vh = self.s @ Vh
		self.Vh = self.Vh[:self.k]


		self.wordCntDoc = wordCntDoc
		self.docs = docs
		self.docIDs = docIDs
		self.idf = idf
		self.index = mat
		self.unq_words = unq_words
		print("K :",self.k)
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
			vec = np.zeros((len(self.unq_words),1))
			cnt = defaultdict(int)

			for sent in query:
				for word in sent:
					# cnt[word] += 1
					# if word.isalpha():
					# 	cnt[word.lower()] += 1
					# else:
					# 	if '-' in word:
					# 		sword = word.lower()
					# 		l = sword.lower().split('-')
					# 		cnt[l[0]] += 1
					# 		cnt[l[1]] += 1
					if word.isalpha():
						if len(wordnet.synsets(word.lower())) > 0:
							cnt[wordnet.synsets(word.lower())[0].name()] += 1
						else:
							cnt[word.lower()] += 1
					else:
						if '-' in word:
							sword = word.lower()
							l = sword.lower().split('-')
							if len(wordnet.synsets(l[0].lower())) > 0:
								cnt[wordnet.synsets(l[0].lower())[0].name()] += 1
							else:
								cnt[l[0].lower()] += 1
							if len(wordnet.synsets(l[1].lower())) > 0:
								cnt[wordnet.synsets(l[1].lower())[0].name()] += 1
							else:
								cnt[l[1].lower()] += 1

			for i, word in enumerate(self.unq_words):
				if cnt[word] > 0:
					vec[i][0] = self.idf[i]*(cnt[word])
					# vec[i][0] = self.idf[i]*(1+np.log(cnt[word]))
				else:
					vec[i][0] = self.idf[i]*(0)
				# if word == "aeroelastic":
				# 	print(self.idf[i], cnt[word])
				# print(word, cnt[word], self.idf[i])
			# scores = np.zeros(self.docIDs[-1]+1)
			scores = [[0, 0] for _ in range(self.docIDs[-1]+1)]
			# print("S shape:",self.s.shape)
			# print("U.T shape:",self.U.T.shape, "vec shape:",vec.shape, "Vh[:,1] shape:",self.Vh[:,1].reshape(self.k,1).shape)
			vec = np.linalg.inv(self.sigma) @ self.U.T @ vec
			# Shape of Vec : (k,k) @ (k,vocab) @ (vocab,1) = (k, 1)
			# Shape of Vh : (k, num_docs), Shape of self.Vh[:,idx] : (k,1)
			# print("Norm of query : ",np.linalg.norm(vec))
			for idx in self.docIDs:
				# sc = np.dot(vec, self.index[:,idx])
				sc = np.dot(vec.T, self.Vh[:,idx].reshape(self.k,1))
				if np.linalg.norm(vec) == 0.0 or np.linalg.norm(self.Vh[:,idx]) == 0.0:
					scores[idx] = [0, idx]
					continue
				sc = sc / np.linalg.norm(vec) 
				sc = sc / np.linalg.norm(self.Vh[:,idx])
				scores[idx] = [sc, idx]
				# scores[idx-1] = [sc, idx, np.linalg.norm(vec), np.linalg.norm(self.index[idx,:]),np.dot(vec, self.index[idx,:])]
			# rem = scores[485]
			scores.sort(reverse=True)
		
			
			# doc_IDs_ordered = []
			order = []
			for sc, idx in scores:
				order.append(idx+1)
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
