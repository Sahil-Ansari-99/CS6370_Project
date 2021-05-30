from util import *
# from util import *
import pandas as pd
from collections import defaultdict
import numpy as np
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
		relevance = np.zeros((len(query_doc_IDs_ordered),1))
		for i in range(len(query_doc_IDs_ordered)):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				relevance[i] = 1

		precision = relevance[:k].sum()/k
		# print(precision)

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

		meanPrecision = 0

		#Fill in code here
		qrels_df = pd.DataFrame(qrels)
		for i in range(len(query_ids)):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			# print("query id :"+str(query_id))
			true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))
			meanPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		meanPrecision /= len(query_ids)
		
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
		relevance = np.zeros((len(query_doc_IDs_ordered),1))
		for i in range(len(query_doc_IDs_ordered)):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				relevance[i] = 1

		recall = relevance[:k].sum()/len(true_doc_IDs)
		#print("In Recall, ", len(true_doc_IDs))

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

		meanRecall = 0

		#Fill in code here
		qrels_df = pd.DataFrame(qrels)
		for i in range(len(query_ids)):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))
			meanRecall += self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		meanRecall /= len(query_ids)

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

		#Fill in code here
		
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		
		fscore = 2*precision*recall/(precision + recall+1e-6)
		
		if (precision +recall) == 0:
			fscore = 0
                        
		


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

		meanFscore = 0

		#Fill in code here
		qrels_df = pd.DataFrame(qrels)
		for i in range(len(query_ids)):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))
			meanFscore += self.queryFscore(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		meanFscore /= len(query_ids)

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
			The list of IDs of documents relevant to the query (ground truth) # some things wrong here, it should be having relevance rating also, so it must be a dict or df
		arg4 : int
			The k value
		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = 0

		#Fill in code here
		relevance = np.zeros((len(query_doc_IDs_ordered), 1))
		true_doc_IDs["position"] = 4-true_doc_IDs["position"] # doubt here 4- or directly


		# finding ideal DCG value
		true_doc_IDs_sorted = true_doc_IDs.sort_values("position", ascending = False)
		DCG_ideal = true_doc_IDs_sorted.iloc[0]["position"]

		for i in range(1, min(k,len(true_doc_IDs))):
			DCG_ideal += true_doc_IDs_sorted.iloc[i]["position"] * np.log(2)/np.log(i+1)

		t_doc_IDs = list(map(int, true_doc_IDs["id"]))
		for i in range(k):
			if query_doc_IDs_ordered[i] in t_doc_IDs:
				relevance[i] = true_doc_IDs[true_doc_IDs["id"] == str(query_doc_IDs_ordered[i])].iloc[0]["position"]

		for i in range(k):
			nDCG += relevance[i] * np.log(2) / np.log(i + 2)  # Note that here index starts from 0

		nDCG = nDCG/DCG_ideal

		# print(nDCG)

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

		meanNDCG = 0

		#Fill in code here
		qrels_df = pd.DataFrame(qrels)
		for i in range(len(query_ids)):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_doc_IDs = qrels_df[["position","id"]][qrels_df["query_num"] == str(query_id)]			
			meanNDCG += self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		meanNDCG /= len(query_ids)


		return meanNDCG[0]


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

		avgPrecision = 0

		#Fill in code here
		relevance = np.zeros((len(query_doc_IDs_ordered), 1))
		for i in range(len(query_doc_IDs_ordered)):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				relevance[i] = 1

		for i in range(min(k,len(relevance))):
			if relevance[i] == 1:
				avgPrecision += self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,i+1)

		if(np.sum(relevance[:k]) == 0):
			return 0
		else:
			return avgPrecision/np.sum(relevance[:k])


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

		meanAveragePrecision = 0

		#Fill in code here
		qrels_df = pd.DataFrame(q_rels)
		for i in range(len(query_ids)):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))
			meanAveragePrecision += self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		meanAveragePrecision /= len(query_ids)


		return meanAveragePrecision