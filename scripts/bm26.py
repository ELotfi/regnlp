import logging
import pandas as pd
import json, random, os
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from stop_words import get_stop_words
import tiktoken

logger = logging.getLogger(__name__)
logging.disable(logging.WARNING)


stop_words = get_stop_words('en')
#nlp = 



import math
import numpy as np
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
	def __init__(self, corpus, tokenizer=None, avgdl_bias=None):
		self.corpus_size = 0
		self.avgdl = 0
		self.doc_freqs = []
		self.idf = {}
		self.doc_len = []
		self.tokenizer = tokenizer
		self.avgdl_bias = avgdl_bias

		if tokenizer:
			corpus = self._tokenize_corpus(corpus)

		nd = self._initialize(corpus)
		self._calc_idf(nd)

	def _initialize(self, corpus):
		nd = {}  # word -> number of documents with word
		num_doc = 0
		for document in corpus:
			self.doc_len.append(len(document))
			num_doc += len(document)

			frequencies = {}
			for word in document:
				if word not in frequencies:
					frequencies[word] = 0
				frequencies[word] += 1
			self.doc_freqs.append(frequencies)

			for word, freq in frequencies.items():
				try:
					nd[word]+=1
				except KeyError:
					nd[word] = 1

			self.corpus_size += 1

		self.avgdl = num_doc / self.corpus_size + self.avgdl_bias
		return nd

	def _tokenize_corpus(self, corpus):
		pool = Pool(cpu_count())
		tokenized_corpus = pool.map(self.tokenizer, corpus)
		return tokenized_corpus

	def _calc_idf(self, nd):
		raise NotImplementedError()

	def get_scores(self, query):
		raise NotImplementedError()

	def get_batch_scores(self, query, doc_ids):
		raise NotImplementedError()

	def get_top_n(self, query, documents, n=5):

		assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

		scores = self.get_scores(query)
		top_n = np.argsort(scores)[::-1][:n]
		return [documents[i] for i in top_n]





class BM25Okapi(BM25):
	def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, avgl_bias=0.):
		self.k1 = k1
		self.b = b
		self.epsilon = epsilon
		self.avgl_bias = avgl_bias
		super().__init__(corpus, tokenizer, avgl_bias)

	def _calc_idf(self, nd):
		"""
		Calculates frequencies of terms in documents and in corpus.
		This algorithm sets a floor on the idf values to eps * average_idf
		"""
		# collect idf sum to calculate an average idf for epsilon value
		idf_sum = 0
		# collect words with negative idf to set them a special epsilon value.
		# idf can be negative if word is contained in more than half of documents
		negative_idfs = []
		for word, freq in nd.items():
			idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
			self.idf[word] = idf
			idf_sum += idf
			if idf < 0:
				negative_idfs.append(word)
		self.average_idf = idf_sum / len(self.idf)

		eps = self.epsilon * self.average_idf
		for word in negative_idfs:
			self.idf[word] = eps

	def get_scores(self, query):
		"""
		The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
		this algorithm also adds a floor to the idf value of epsilon.
		See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
		:param query:
		:return:
		"""
		score = np.zeros(self.corpus_size)
		doc_len = np.array(self.doc_len)
		for q in query:
			q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
			score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
											   (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
		return score

	def get_batch_scores(self, query, doc_ids):
		"""
		Calculate bm25 scores between query and subset of all docs
		"""
		assert all(di < len(self.doc_freqs) for di in doc_ids)
		score = np.zeros(len(doc_ids))
		doc_len = np.array(self.doc_len)[doc_ids]
		for q in query:
			q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
			score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
											   (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
		return score.tolist()






class BM25Mut():
	def __init__(self, args):
		self.args = args 
		self.encoder = tiktoken.encoding_for_model("gpt-4") # if args.bm25_tok == 'gpt4' else spacy.load("en_core_news_sm") if args.bm25_tok == 'stem' else None
		self.corpus = json.load(open(args.corpus_path))
		self.queries = json.load(open(args.train_path))
		self.idx = json.load(open(args.index_path))
		self.build_mutindex()



	def encode(self, text):
		text = text.lower()
		if self.args.remove_stop_words:
			text = ' '.join([w for w in text.split() if w not in stop_words])
		tokens = self.encoder.encode(text)
		return [str(token) for token in tokens]




	def build_mutindex(self):
		print('Building Mutual Index...')
		tok_docs, tok_qurs = [], []
		for _,v in tqdm(self.corpus.items()):
			tok_docs.append(self.encode(v['txt'])) 		
		for q in self.queries:
			tok_qurs.append(self.encode(q['query']))
		doc_index = BM25Okapi(tok_docs, k1=self.args.k1, b=self.args.b, avgl_bias=self.args.avgdl_bias)
		qur_index = BM25Okapi(tok_qurs, k1=self.args.k1, b=self.args.b, avgl_bias=self.args.avgdl_bias)
		self.doc_idf, self.q_idf = doc_index.idf, qur_index.idf
		self.corpus_size = doc_index.corpus_size
		self.doc_len = doc_index.doc_len
		self.doc_freqs = doc_index.doc_freqs
		self.avgdl = doc_index.avgdl
		


	def get_scores(self, query):
		score = np.zeros(self.corpus_size)
		doc_len = np.array(self.doc_len)
		for q in query:
			idf_score = self.args.didf_c * self.doc_idf.get(q, 0) + self.args.qidf_c * self.q_idf.get(q, 0)
			q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
			score += (idf_score) * (q_freq * (self.args.k1 + 1) /
											   (q_freq + self.args.k1 * (1 - self.args.b + self.args.b * doc_len / self.avgdl)))
		return score



	def get_topk(self, query, k):
		query_tok = self.encode(query)
		scores = self.get_scores(query_tok)
		scores = [[s,i] for i,s in enumerate(scores.tolist())]
		scores = sorted(scores, reverse=True)
		return np.array([sc[1] for sc in scores[:k]])


	def search(self, queries, k):
		indices = []
		for q in tqdm(queries):
			topk = self.get_topk(q['query'], k)
			indices.append(topk)
		return indices



###########################################################


class BM25Ret():
	def __init__(self, args) -> None:
		self.args = args
		self.encoder = tiktoken.encoding_for_model("gpt-4") # if args.bm25_tok == 'gpt4' else spacy.load("en_core_news_sm") if args.bm25_tok == 'stem' else None
		self.corpus = json.load(open(args.corpus_path))
		self.idx = json.load(open(args.index_path))
		self.build_bm25index()



	def encode(self, text):
		text = text.lower()
		if self.args.remove_stop_words:
			text = ' '.join([w for w in text.split() if w not in stop_words])
		tokens = self.encoder.encode(text)
		return [str(token) for token in tokens]




	def build_bm25index(self):
		print('Building Index...')
		tok_corpus = []
		for _,v in tqdm(self.corpus.items()):
			tok_corpus.append(self.encode(v['txt'])) 		
		self.index = BM25Okapi(tok_corpus, k1=self.args.k1, b=self.args.b, avgl_bias=self.args.avgdl_bias)



	def get_topk(self, query, k):
		query_tok = self.encode(query)
		scores = self.index.get_scores(query_tok)
		scores = [[s,i] for i,s in enumerate(scores.tolist())]
		scores = sorted(scores, reverse=True)
		return np.array([sc[1] for sc in scores[:k]])


	def search(self, queries, k):
		indices = []
		for q in tqdm(queries):
			topk = self.get_topk(q['query'], k)
			indices.append(topk)
		return indices
