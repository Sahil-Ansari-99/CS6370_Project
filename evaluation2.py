from util import *
from collections import defaultdict
import numpy as np
import math
from math import log
# Add your import statements here


class Evaluation:

	def __init__(self):
		self.qrels = None

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
		precisions = []
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			curr = self.queryPrecision(pred_ids, query_ids[i], true_ids, k)
			p += curr
			precisions.append(curr)
		meanPrecision = p / n
		return meanPrecision, precisions

	
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
		recalls = []
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			curr = self.queryRecall(pred_ids, query_ids[i], true_ids, k)
			p += curr
			recalls.append(curr)
		meanRecall = p / n

		return meanRecall, recalls


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
		fscores = []
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			curr = self.queryFscore(pred_ids, query_ids[i], true_ids, k)
			p += curr
			fscores.append(curr)
		meanFscore = p / n

		return meanFscore, fscores
	

	# def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
	# 	"""
	# 	Computation of nDCG of the Information Retrieval System
	# 	at given value of k for a single query

	# 	Parameters
	# 	----------
	# 	arg1 : list
	# 		A list of integers denoting the IDs of documents in
	# 		their predicted order of relevance to a query
	# 	arg2 : int
	# 		The ID of the query in question
	# 	arg3 : list
	# 		The list of IDs of documents relevant to the query (ground truth)
	# 	arg4 : int
	# 		The k value

	# 	Returns
	# 	-------
	# 	float
	# 		The nDCG value as a number between 0 and 1
	# 	"""
	# 	# qrels = true_doc_IDs
	# 	# rels = {}
	# 	# for qrel in qrels:
	# 	# 	if int(qrel["query_num"]) == query_id:
	# 	# 		rels[int(qrel["id"])] = 5 - int(qrel["position"])

	# 	# nax = min(k, len(rels))
	# 	# dcg = 0
	# 	# for i, doc_id in enumerate(query_doc_IDs_ordered[:nax]):
	# 	# 	if doc_id in rels:
	# 	# 		dcg += rels[doc_id]/np.log2(i+2)
		
	# 	# ideal = list(rels.values)
	# 	# ideal.sort(reverse=True)
	# 	# idcg = 0
	# 	# for i, rel in enumerate(ideal[:k]):
	# 	# 	idcg += rel/np.log2(i+2)
	# 	# return dcg/idcg


	# 	nDCG = -1
	# 	order = query_doc_IDs_ordered
	# 	#Fill in code here
	# 	dcg = 0
	# 	true_docs = [x for x, y in true_doc_IDs]
	# 	true_rels = [y for x, y in true_doc_IDs]
	# 	# print("True Rels Length : ",len(true_rels))
	# 	# print("True Docs Length : ",len(true_docs))

	# 	nax = min(k, len(order))
	# 	for i in range(nax):
	# 		if order[i] in true_docs:
	# 			idx = true_docs.index(order[i])
	# 			dcg += (5-true_rels[idx])/np.log2(i+2)

	# 	iorder = []
	# 	true_order = [[y,x] for x, y in true_doc_IDs]
	# 	true_order.sort()
	# 	for i in range(min(k, len(true_order))):
	# 		iorder.append(5-true_order[i][0])
				
	# 	iorder.sort(reverse=True)
	# 	idcg = 0
	# 	for i in range(min(k, len(iorder))):
	# 		idcg += iorder[i]/np.log2(i+2)

	# 	if idcg == 0:
	# 		nDCG = 0
	# 	else:
	# 		nDCG = dcg / idcg
	# 	return nDCG


	# def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
	# 	"""
	# 	Computation of nDCG of the Information Retrieval System
	# 	at a given value of k, averaged over all the queries

	# 	Parameters
	# 	----------
	# 	arg1 : list
	# 		A list of lists of integers where the ith sub-list is a list of IDs
	# 		of documents in their predicted order of relevance to the ith query
	# 	arg2 : list
	# 		A list of IDs of the queries for which the documents are ordered
	# 	arg3 : list
	# 		A list of dictionaries containing document-relevance
	# 		judgements - Refer cran_qrels.json for the structure of each
	# 		dictionary
	# 	arg4 : int
	# 		The k value

	# 	Returns
	# 	-------
	# 	float
	# 		The mean nDCG value as a number between 0 and 1
	# 	"""

	# 	meanNDCG = -1
	# 	# self.qrels = qrels
	# 	#Fill in code here
	# 	dic = defaultdict(list)
	
	# 	for d in qrels:
	# 		# print("D: ",d)
	# 		dic[int(d["query_num"])].append([int(d['id']), d['position']])

	# 	p = 0
	# 	n = len(query_ids)
	# 	# pp = 0
	# 	# for i, qid in enumerate(query_ids):

	# 	# 	pp += self.queryNDCG(doc_IDs_ordered[i], qid, qrels,k)
	# 	for i in range(n):
	# 		true_ids = dic[int(query_ids[i])]
	# 		pred_ids = doc_IDs_ordered[i]
	# 		p += self.queryNDCG(pred_ids, query_ids[i], true_ids, k)
	# 	meanNDCG = p / n
	# 	# print("My:",p/n)

	# 	return meanNDCG

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

		#Fill in code here
		qrels=self.qrels
		nDCG, dcg, idcg = 0.0, 0.0, 0.0

		relevance={}
		for qrel in qrels:
			if int(qrel["query_num"]) == query_id:
				relevance[int(qrel["id"])] = 5 - int(qrel["position"])

		# ideal_rank = list(relevance.values())
		# ideal_rank.sort(reverse=True)

		kmax=min(k,len(relevance))
		obs_rank=[]
		for i,doc_id in enumerate(query_doc_IDs_ordered[:kmax]):
			if doc_id in relevance: 
				dcg += relevance[doc_id]/log(i+2, 2) # log(i+1 + 1) as i runs from 0
				obs_rank.append(relevance[doc_id])
			# idcg += ideal_rank[i]/log(i+2, 2)
		# obs_rank.sort(reverse=True)
		
		ideal_rank = list(relevance.values())
		ideal_rank.sort(reverse=True)
		for i, rel in enumerate(ideal_rank[:k]):
			idcg += rel/log(i+2, 2)

		if dcg==0.0 : 
			nDCG=0.0
		else: 
			nDCG = dcg/idcg
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
		if self.qrels is None: self.qrels = qrels

		meanNDCG = 0.0
		dcgs = []
		for i, query_id in enumerate(query_ids):
			curr = self.queryNDCG(doc_IDs_ordered[i], query_id, None, k)
			meanNDCG += curr
			dcgs.append(curr)

		meanNDCG = meanNDCG/len(query_ids)
		return meanNDCG, dcgs


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
		maps = []
		for i in range(n):
			true_ids = dic[int(query_ids[i])]
			pred_ids = doc_IDs_ordered[i]
			curr = self.queryAveragePrecision(pred_ids, query_ids[i], true_ids, k)
			p += curr
			maps.append(curr)
		meanAveragePrecision = p / n


		return meanAveragePrecision, maps