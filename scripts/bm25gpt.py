import json
import numpy as np
from tqdm import tqdm
from rankbm25 import BM25Okapi as bm25
from stop_words import get_stop_words
import tiktoken
from multiprocessing import Pool, cpu_count



stop_words = get_stop_words('en')

class BM25():
	def __init__(self, args, corpus=None, k1=1.5, b=.75) -> None:
		self.args = args
		self.encoder = tiktoken.encoding_for_model("gpt-4") 
		self.corpus = json.load(open(args.corpus_path)) if corpus is None else corpus
		self.k1 = k1
		self.b = b
		self.build_bm25index()



	def encode(self, text, is_query=True):
		text = text.lower()
		if self.args.remove_stop_words:
			text = ' '.join([w for w in text.split() if w not in stop_words])
		tokens = self.encoder.encode(text)
		if is_query: tokens = [t for t in tokens if str(t) in self.voc]
		return [str(token) for token in tokens]




	def build_bm25index(self):
		print('Building Index...')
		tok_corpus = []
		for _,v in tqdm(self.corpus.items()):
			tok_corpus.append(self.encode(v['txt'], False))
		self.voc = list(set([t for tk in tok_corpus for t in tk])) 		
		self.index = bm25(tok_corpus, k1=self.k1, b=self.b)



	def get_topk(self, query, k=100):
		query_tok = self.encode(query)
		scores = self.index.get_scores(query_tok)
		scores = [[s,i] for i,s in enumerate(scores.tolist())]
		scores = sorted(scores, reverse=True)
		return np.array([sc[1] for sc in scores[:k]])


	def search(self, queries):
		indices = []
		pool = Pool(cpu_count())
		indices = pool.map(self.get_topk, queries)
		return indices


