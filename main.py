from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval_vsm import InformationRetrieval
from informationRetrieval2 import InformationRetrieval as InformationRetrieval_LSA
from ir_1 import InformationRetrieval as InformationRetrieval_Base
from informationRetrieval_base_lsa import InformationRetrieval as InformationRetrieval_Base_LSA
from informationRetrieval_lsa_qe import InformationRetrieval as InformationRetrieval_LSA_QE
from evaluation2 import Evaluation
from scipy.stats import norm
import numpy as np
import time

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval()
		self.informationRetriever_lsa = InformationRetrieval_LSA()
		self.informationRetriever_base = InformationRetrieval_Base()
		self.informationRetriever_base_lsa = InformationRetrieval_Base_LSA()
		self.informationRetriever_lsa_qe = InformationRetrieval_LSA_QE()
		self.evaluator = Evaluation()

	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
		# titles = [item["title"]*3 for item in docs_json]
		# docs = titles+docs
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		cluster_sizes = [20, 50, 70, 100, 150, 200, 250]
		ks = [50, 100, 200, 300, 500]
		map_at_5 = []
		map_at_10 = []
		fscore_at_5 = []
		fscore_at_10 = []
		precisions_at_5 = []
		recalls_at_5 = []
		fscores_at_5 = []
		dcgs_at_5 = []
		maps_at_5 = []
		precisions_at_5_lsa = []
		recalls_at_5_lsa = []
		fscores_at_5_lsa = []
		dcgs_at_5_lsa = []
		maps_at_5_lsa = []

		self.informationRetriever_base.buildIndex(processedDocs, doc_ids)
		# self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# self.informationRetriever_lsa.buildIndex(processedDocs, doc_ids)
		# self.informationRetriever_base_lsa.buildIndex(processedDocs, doc_ids)
		self.informationRetriever_lsa_qe.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		start_time = time.time()
		# doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
		# doc_IDs_ordered_lsa = self.informationRetriever_lsa.rank(processedQueries)
		doc_IDs_ordered_base = self.informationRetriever_base.rank(processedQueries)
		doc_IDs_ordered_base_lsa = self.informationRetriever_lsa_qe.rank(processedQueries)
		end_time = time.time()
		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision, precision_list = self.evaluator.meanPrecision(
				doc_IDs_ordered_base, query_ids, qrels, k)
			precision_lsa, precision_list_lsa = self.evaluator.meanPrecision(
				doc_IDs_ordered_base_lsa, query_ids, qrels, k)
			precisions.append(precision)
			recall, recall_list = self.evaluator.meanRecall(
				doc_IDs_ordered_base, query_ids, qrels, k)
			recall_lsa, recall_list_lsa = self.evaluator.meanRecall(
				doc_IDs_ordered_base_lsa, query_ids, qrels, k)
			recalls.append(recall)
			fscore, fscore_list = self.evaluator.meanFscore(
				doc_IDs_ordered_base, query_ids, qrels, k)
			fscore_lsa, fscore_list_lsa = self.evaluator.meanFscore(
				doc_IDs_ordered_base_lsa, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +
				str(k) + " : " + str(precision) + ", " + str(recall) +
				", " + str(fscore))
			MAP, maps_list = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered_base, query_ids, qrels, k)
			MAP_lsa, maps_list_lsa = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered_base_lsa, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG, dcg_list = self.evaluator.meanNDCG(
				doc_IDs_ordered_base, query_ids, qrels, k)
			nDCG_lsa, dcg_list_lsa = self.evaluator.meanNDCG(
				doc_IDs_ordered_base_lsa, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " + str(k) + " : " + str(MAP) + ", " + str(nDCG))
			print("MAP, nDCG @ " + str(k) + " : " + str(MAP_lsa) + ", " + str(nDCG_lsa))
			if k == 10:
				precisions_at_5 = precision_list
				recalls_at_5 = recall_list
				fscores_at_5 = fscore_list
				dcgs_at_5 = dcg_list
				maps_at_5 = maps_list
				precisions_at_5_lsa = precision_list_lsa
				recalls_at_5_lsa = recall_list_lsa
				fscores_at_5_lsa = fscore_list_lsa
				dcgs_at_5_lsa = dcg_list_lsa
				maps_at_5_lsa = maps_list_lsa
		# fscore = self.evaluator.meanFscore(
		# 	doc_IDs_ordered, query_ids, qrels, 5)
		# fscore_at_5.append(fscore)
		# fscore = self.evaluator.meanFscore(
		# 	doc_IDs_ordered, query_ids, qrels, 10)
		# fscore_at_10.append(fscore)
		# MAP = self.evaluator.meanAveragePrecision(
		# 	doc_IDs_ordered, query_ids, qrels, 5)
		# map_at_5.append(MAP)
		# MAP = self.evaluator.meanAveragePrecision(
		# 	doc_IDs_ordered, query_ids, qrels, 10)
		# map_at_10.append(MAP)
		# Plot the metrics and save plot
		# plt.plot(range(1, 11), precisions, label="Precision")
		# plt.plot(range(1, 11), recalls, label="Recall")
		# plt.plot(range(1, 11), fscores, label="F-Score")
		# plt.plot(range(1, 11), MAPs, label="MAP")
		# plt.plot(range(1, 11), nDCGs, label="nDCG")
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset")
		# plt.xlabel("k")
		# print("-"*50)
		# print("PLOTTING")
		# print("-"*50)
		# plt.savefig(args.out_folder + "eval_plot_normal.png")
		# print('\n')
		mu, std = norm.fit(dcgs_at_5)
		mu_lsa, std_lsa = norm.fit(dcgs_at_5_lsa)
		fig1, ax1 = plt.subplots()
		ax1.hist(dcgs_at_5, bins=10, histtype='stepfilled', color='royalblue', alpha=0.8, label='VSM', density=True)
		ax1.hist(dcgs_at_5_lsa, bins=10, histtype='stepfilled', color='lightcoral', alpha=0.8, label='LSA with QE', density=True)
		xmin, xmax = ax1.get_xlim()
		x = np.linspace(xmin, xmax, 100)
		p = norm.pdf(x, mu, std)
		p_lsa = norm.pdf(x, mu_lsa, std_lsa)
		ax1.plot(x, p, color='b', linewidth=2)
		ax1.plot(x, p_lsa, color='r', linewidth=2)
		ax1.set_xlabel('nDCG @ k=10')
		ax1.set_ylabel('Number of Queries')
		ax1.legend()
		plt.show()
		# self.plot_histogram(recalls_at_5, recalls_at_5_lsa, 'Recall')
		# self.plot_histogram(fscore_at_5, fscores_at_5_lsa, 'F-Score')
		# self.plot_histogram(dcgs_at_5, dcgs_at_5_lsa, 'nDCG')
		print('Time taken:', end_time - start_time, 'seconds')
		# fig1, ax1 = plt.subplots()
		# ax1.plot(ks, fscore_at_5, label="@5")
		# ax1.plot(ks, fscore_at_10, label="@10")
		# ax1.legend()
		# ax1.set_xlabel("Reduced Dimension")
		# ax1.set_ylabel("F-Score")
		# ax1.set_title("Evaluation Metrics - Cranfield Dataset")
		# fig2, ax2 = plt.subplots()
		# ax2.plot(ks, map_at_5, label="@5")
		# ax2.plot(ks, map_at_10, label="@10")
		# ax2.legend()
		# ax2.set_xlabel("Reduced Dimension")
		# ax2.set_ylabel("MAP")
		# ax2.set_title("Evaluation Metrics - Cranfield Dataset")
		# plt.show()

	def plot_histogram(self, data1, data2, name):
		mu, std = norm.fit(data1)
		mu_lsa, std_lsa = norm.fit(data2)
		fig1, ax1 = plt.subplots()
		ax1.hist(data1, bins=10, histtype='stepfilled', color='royalblue', alpha=0.8, label='VSM with QE', density=True)
		ax1.hist(data2, bins=10, histtype='stepfilled', color='lightcoral', alpha=0.8, label='VSM', density=True)
		xmin, xmax = ax1.get_xlim()
		x = np.linspace(xmin, xmax, 100)
		p = norm.pdf(x, mu, std)
		p_lsa = norm.pdf(x, mu_lsa, std_lsa)
		ax1.plot(x, p, color='b', linewidth=2)
		ax1.plot(x, p_lsa, color='r', linewidth=2)
		ax1.set_xlabel(name + ' @ k=10')
		ax1.set_ylabel('Number of Queries')
		ax1.legend()
		filename_ = name + 'k_10_vsm_vs_vsm_qe.png'
		fig1.savefig('plots/' + filename_)

	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
