from util import *
from collections import defaultdict
import numpy as np
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		ordered = query_doc_IDs_ordered
		cnt = 0
		for doc in ordered[:k]:
			if doc in true_doc_IDs:
				cnt += 1
		
		cnt /= k
		precision = cnt

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
		# print(qrels[1], len(qrels))
		meanPrecision = -1
		dic = defaultdict(list)
		for d in qrels:
			dic[int(d["query_num"])].append(int(d['id']))
		#Fill in code here
		p = 0
		n = len(query_ids)
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			p += self.queryPrecision(pred_ids, query_ids[i], true_ids, k)
		meanPrecision = p / n
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		ordered = query_doc_IDs_ordered
		cnt = 0
		for doc in ordered[:k]:
			if doc in true_doc_IDs:
				cnt += 1
		
		cnt /= len(true_doc_IDs)
		recall = cnt


		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		dic = defaultdict(list)
		for d in qrels:
			dic[int(d["query_num"])].append(int(d['id']))
		p = 0
		n = len(query_ids)
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			p += self.queryRecall(pred_ids, query_ids[i], true_ids, k)
		meanRecall = p / n

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = (2*precision*recall) / (precision+recall+1e-6)
		#Fill in code here

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		dic = defaultdict(list)
		for d in qrels:
			dic[int(d["query_num"])].append(int(d['id']))
		p = 0
		n = len(query_ids)
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			p += self.queryFscore(pred_ids, query_ids[i], true_ids, k)
		meanFscore = p / n


		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1
		order = query_doc_IDs_ordered
		#Fill in code here
		dcg = 0
		true_docs = [x for x, y in true_doc_IDs]
		true_rels = [y for x, y in true_doc_IDs]
		# print("True Rels Length : ",len(true_rels))
		# print("True Docs Length : ",len(true_docs))
		
		for i in range(k):
			if order[i] in true_docs:
				idx = true_docs.index(order[i])
				dcg += (5-true_rels[idx])/np.log2(i+2)

		iorder = []
		for i in range(k):
			if order[i] in true_docs:
				idx = true_docs.index(order[i])
				iorder.append(5-true_rels[idx])
			else:
				iorder.append(0)
		iorder.sort(reverse=True)
		idcg = 0
		for i in range(k):
			idcg += iorder[i]/np.log2(i+2)
		# print("DCG", dcg, "IDCG", idcg)
		if idcg == 0:
			nDCG = 0
		else:
			nDCG = dcg / idcg
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		dic = defaultdict(list)
		for d in qrels:
			dic[int(d["query_num"])].append([int(d['id']), d['position']])

		p = 0
		n = len(query_ids)
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			p += self.queryNDCG(pred_ids, query_ids[i], true_ids, k)
		meanNDCG = p / n


		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		order = query_doc_IDs_ordered
		cnt, p = 0, 0
		for i in range(k):
			if order[i] in true_doc_IDs:
				cnt += 1
				p += cnt/(i+1)
		if cnt == 0:
			avgPrecision = 0
		else:
			avgPrecision = p / cnt

	

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		dic = defaultdict(list)
		for d in q_rels:
			dic[int(d["query_num"])].append(int(d['id']))
		p = 0
		n = len(query_ids)
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			p += self.queryAveragePrecision(pred_ids, query_ids[i], true_ids, k)
		meanAveragePrecision = p / n


		return meanAveragePrecision