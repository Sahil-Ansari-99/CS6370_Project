from util import *
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet
from sklearn.cluster import KMeans
from sklearn import preprocessing


# Add your import statements here


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.idfs = None
        self.tf_idfs = None
        self.synsets_count = 0
        self.synsets_idx = None
        self.unique_synsets = None
        self.u_mat = None
        self.s_mat = None
        self.v_mat = None
        self.s_inv = None
        self.doc_ids = None
        self.k = 300
        self.clusters = 200
        self.docs_to_retrieve = 100
        self.cluster_centers = None
        self.cluster_docs = None

    def set_cluster_size(self, num_clusters):
        self.clusters = num_clusters

    def set_k(self, new_k):
        self.k = new_k

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

        # Fill in code here
        # print(docs)
        self.doc_ids = docIDs
        N = docIDs[-1]

        words_synsets = []
        known = 0
        unknown = 0
        for doc in docs:
            for sentence in doc:
                for word in sentence:
                    if not word.isalpha():
                        if '-' in word:
                            curr_synsets = wordnet.synsets(word.lower())
                            if len(curr_synsets) > 0:
                                words_synsets.append(curr_synsets[0].name())
                            else:
                                word1, word2 = word.split('-')[0], word.split('-')[1]
                                syn1 = wordnet.synsets(word1.lower())
                                syn2 = wordnet.synsets(word2.lower())
                                if len(syn1) > 0:
                                    words_synsets.append(syn1[0].name())
                                else:
                                    words_synsets.append(word1)
                                    unknown += 1
                                if len(syn2) > 0:
                                    words_synsets.append(syn2[0].name())
                                else:
                                    words_synsets.append(word2)
                                    unknown += 1
                    else:
                        curr_synsets = wordnet.synsets(word.lower())
                        if len(curr_synsets) > 0:
                            words_synsets.append(curr_synsets[0].name())
                        else:
                            words_synsets.append(word.lower())
                            unknown += 1
        unique_synsets = list(set(words_synsets))
        self.unique_synsets = unique_synsets
        self.synsets_count = len(unique_synsets)
        print(len(unique_synsets), unknown)
        words = []
        unq_words = list(set(words))
        unique_synsets_idx = {}
        for i, synset in enumerate(unique_synsets):
            unique_synsets_idx[synset] = i
        m = docIDs[-1]
        n = len(unique_synsets_idx)
        self.synsets_idx = unique_synsets_idx
        print('----------------------------------------------')
        print('Unique Synsets Acquired')
        print('----------------------------------------------')
        # print("Total Number of Unique Words : {}".format(len(unq_words)))

        # wordCntDoc = [defaultdict(int) for _ in range(docIDs[-1] + 1)]
        # for idx in docIDs:
        #     for sent in docs[idx - 1]:
        #         for word in sent:
        #             wordCntDoc[idx][word] += 1
        synsetDocFreq = np.zeros((m, n))
        synsetCount = defaultdict(int)
        for i, doc in enumerate(docs):
            for sentence in doc:
                for word in sentence:
                    if not word.isalpha():
                        if '-' in word:
                            curr_synsets = wordnet.synsets(word.lower())
                            if len(curr_synsets) > 0:
                                synsetDocFreq[i][unique_synsets_idx.get(curr_synsets[0].name())] += 1
                                synsetCount[curr_synsets[0].name()] += 1
                            else:
                                word1, word2 = word.split('-')[0], word.split('-')[1]
                                syn1 = wordnet.synsets(word1.lower())
                                syn2 = wordnet.synsets(word2.lower())
                                if len(syn1) > 0:
                                    synsetDocFreq[i][unique_synsets_idx.get(syn1[0].name())] += 1
                                    synsetCount[syn1[0].name()] += 1
                                else:
                                    synsetDocFreq[i][unique_synsets_idx.get(word1)] += 1
                                    synsetCount[word1] += 1
                                if len(syn2) > 0:
                                    synsetDocFreq[i][unique_synsets_idx.get(syn2[0].name())] += 1
                                    synsetCount[syn2[0].name()] += 1
                                else:
                                    synsetDocFreq[i][unique_synsets_idx.get(word2)] += 1
                                    synsetCount[word2] += 1
                    else:
                        curr_synsets = wordnet.synsets(word.lower())
                        if len(curr_synsets) > 0:
                            synsetDocFreq[i][unique_synsets_idx.get(curr_synsets[0].name())] += 1
                            synsetCount[curr_synsets[0].name()] += 1
                        else:
                            synsetDocFreq[i][unique_synsets_idx.get(word.lower())] += 1
                            synsetCount[word] += 1
            # freq_sum = np.sum(synsetDocFreq[i])
            for idx, freq in enumerate(synsetDocFreq[i]):
                synsetDocFreq[i][idx] = np.log(1 + freq)

        print("----------------------------------------------")
        print("Built Term Frequency Doc")
        print("----------------------------------------------")
        idfs = defaultdict(int)
        wordCntDoc = []
        for doc in synsetDocFreq:
            for synset in unique_synsets:
                if doc[unique_synsets_idx.get(synset)] > 0:
                    idfs[synset] += 1
        for synset in idfs:
            idfs[synset] = np.log(n / idfs.get(synset))
        self.idfs = idfs
        print("----------------------------------------------")
        print("Built IDFs")
        print("----------------------------------------------")

        mat = np.zeros((N + 1, len(unq_words)))
        for doc in synsetDocFreq:
            for synset in unique_synsets:
                doc[unique_synsets_idx.get(synset)] *= idfs.get(synset)

        print("----------------------------------------------")
        print("Built TF-IDF matrix")
        print("----------------------------------------------")

        u, s, v = np.linalg.svd(synsetDocFreq)
        print(u.shape, s.shape, v.shape)
        s = s[:self.k]
        s_mat = np.zeros((self.k, self.k))
        for i, x in enumerate(s):
            s_mat[i][i] = x
        u_mat = u[:, :self.k]
        v_mat = v[:self.k]
        print(u_mat.shape, s_mat.shape, v_mat.shape)
        latent_mat = np.matmul(u_mat, np.matmul(s_mat, v_mat))
        u_mat = np.matmul(u_mat, s_mat)
        # u_mat = preprocessing.normalize(u_mat)
        self.u_mat = u_mat
        self.s_mat = s_mat
        self.s_inv = np.linalg.inv(s_mat)
        self.v_mat = v_mat
        self.index = latent_mat
        print(self.index.shape)
        # kmeans = KMeans(n_clusters=self.clusters, random_state=42).fit(self.u_mat)
        # assigned_docs = {}
        # for i, center in enumerate(kmeans.labels_):
        #     if assigned_docs.get(center) is None:
        #         assigned_docs[center] = []
        #     assigned_docs.get(center).append(i)
        small_clusters = {}
        # for center in assigned_docs:
        #     print(len(assigned_docs.get(center)))
        # self.cluster_centers = preprocessing.normalize(list(kmeans.cluster_centers_))
        # for center in assigned_docs:
        #     cluster_points = assigned_docs.get(center)
        #     if len(cluster_points) < 10:
        #         small_clusters[center] = cluster_points
        # for center in small_clusters:
        #     cluster_points = small_clusters.get(center)
        #     assigned_docs.pop(center)
        #     for point in cluster_points:
        #         min_dist = 99999999
        #         new_center = None
        #         for big_cluster in assigned_docs:
        #             curr_dist = np.linalg.norm(point - self.cluster_centers[big_cluster])
        #             if curr_dist < min_dist:
        #                 min_dist = curr_dist
        #                 new_center = big_cluster
        #         assigned_docs.get(new_center).append(point)
        # self.cluster_docs = assigned_docs
        # print("----------------------------------------------")
        # print("Built latent matrix and clustering complete...")
        # print("----------------------------------------------")
        # ones = 0
        # for center in assigned_docs:
        #     print(len(assigned_docs.get(center)))
        #     if len(assigned_docs.get(center)) == 1:
        #         ones += 1
        # print(ones)
        return
        # self.wordCntDoc = wordCntDoc
        # self.docs = docs
        # self.docIDs = docIDs
        # self.idf = idf
        # self.index = mat
        # self.unq_words = unq_words

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
            vec = np.zeros(self.synsets_count)
            for sentence in query:
                for word in sentence:
                    if not word.isalpha():
                        if '-' in word:
                            curr_synsets = wordnet.synsets(word.lower())
                            if len(curr_synsets) > 0 and self.synsets_idx.get(curr_synsets[0].name()) is not None:
                                vec[self.synsets_idx.get(curr_synsets[0].name())] += 1
                            else:
                                word1, word2 = word.split('-')[0], word.split('-')[1]
                                syn1 = wordnet.synsets(word1.lower())
                                syn2 = wordnet.synsets(word2.lower())
                                if len(syn1) > 0 and self.synsets_idx.get(syn1[0].name()) is not None:
                                    vec[self.synsets_idx.get(syn1[0].name())] += 1
                                else:
                                    if self.synsets_idx.get(word1) is not None:
                                        vec[self.synsets_idx.get(word1)] += 1
                                if len(syn2) > 0 and self.synsets_idx.get(syn2[0].name()) is not None:
                                    vec[self.synsets_idx.get(syn2[0].name())] += 1
                                elif self.synsets_idx.get(word2) is not None:
                                    vec[self.synsets_idx.get(word2)] += 1
                    else:
                        curr_synsets = wordnet.synsets(word.lower())
                        if len(curr_synsets) > 0 and self.synsets_idx.get(curr_synsets[0].name()) is not None:
                            vec[self.synsets_idx.get(curr_synsets[0].name())] += 1
                        elif self.synsets_idx.get(word) is not None:
                            vec[self.synsets_idx.get(word)] += 1
            for synset in self.unique_synsets:
                vec[self.synsets_idx.get(synset)] *= self.idfs.get(synset)
            vec = np.matmul(vec, np.matmul(self.v_mat.T, self.s_inv))
            vec = vec / np.linalg.norm(vec)
            scores = []
            for idx, doc in enumerate(self.u_mat):
                score = np.dot(vec, doc)
                if np.linalg.norm(vec) == 0 or np.linalg.norm(doc) == 0:
                    scores.append((idx, 0))
                    continue
                # score /= (np.linalg.norm(vec) * np.linalg.norm(center))
                scores.append((idx, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            order = []
            for idx, center in scores:
                order.append(idx + 1)
            doc_IDs_ordered.append(order)
        # Fill in code here

        return doc_IDs_ordered
