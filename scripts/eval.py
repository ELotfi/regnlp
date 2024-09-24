import faiss
import torch
import logging
import pandas as pd
import json, random, os
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

from rankbm25 import BM25Okapi as bm25
from bm26 import BM25Ret as BM26Ret
from bm26 import BM25Mut
from rankbm25_ml import BM25TnInfer
from ranker import RankerInfer
from stop_words import get_stop_words
import tiktoken
import spacy

logger = logging.getLogger(__name__)
logging.disable(logging.WARNING)


stop_words = get_stop_words('en')
#nlp = 





class BM25Ret():
	def __init__(self, args) -> None:
		self.args = args
		self.encoder = tiktoken.encoding_for_model("gpt-4") if args.bm25_tok == 'gpt4' else spacy.load("en_core_news_sm") if args.bm25_tok == 'stem' else None
		self.corpus = json.load(open(args.corpus_path))
		self.idx = json.load(open(args.index_path))
		self.build_bm25index()



	def encode(self, text):
		text = text.lower()
		if self.args.remove_stop_words:
			text = ' '.join([w for w in text.split() if w not in stop_words])
		if self.args.bm25_tok == 'split':
			return text.split()
		elif self.args.bm25_tok == 'gpt4':
			tokens = self.encoder.encode(text)
			return [str(token) for token in tokens]
		elif self.args.bm25_tok == 'stem':
			return [tok.lemma_ for tok in self.encoder(text)]



	def build_bm25index(self):
		print('Building Index...')
		tok_corpus = []
		for _,v in tqdm(self.corpus.items()):
			tok_corpus.append(self.encode(v['txt'])) 		
		self.index = bm25(tok_corpus, k1=self.args.k1, b=self.args.b)



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




def dense_index(model, corpus: list, args):
	"""
	1. Encode the entire corpus into dense embeddings; 
	2. Create faiss index; 
	3. Optionally save embeddings.
	"""

	corpus_embeddings = model.encode_corpus(corpus)
	dim = corpus_embeddings.shape[-1]
	# create faiss index
	faiss_index = faiss.index_factory(dim, args.index_factory, faiss.METRIC_INNER_PRODUCT)

	if args.device == torch.device("cuda"):
		# co = faiss.GpuClonerOptions()
		co = faiss.GpuMultipleClonerOptions()
		co.useFloat16 = True
		# faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
		faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

	# NOTE: faiss only accepts float32
	logger.info("Adding embeddings...")
	corpus_embeddings = corpus_embeddings.astype(np.float32)
	faiss_index.train(corpus_embeddings)
	faiss_index.add(corpus_embeddings)
	return faiss_index


def search(model, queries: list, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
	"""
	1. Encode queries into dense embeddings;
	2. Search through faiss index
	"""
	qurs = [q['query'] for q in queries]
	query_embeddings = model.encode_queries(qurs, batch_size=batch_size, max_length=max_length)
	query_size = len(query_embeddings)
	
	all_scores = []
	all_indices = []
	
	for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
		j = min(i + batch_size, query_size)
		query_embedding = query_embeddings[i: j]
		score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
		all_scores.append(score)
		all_indices.append(indice)
	
	all_scores = np.concatenate(all_scores, axis=0)
	all_indices = np.concatenate(all_indices, axis=0)
	return all_scores, all_indices
	



def evaluate(preds, labels, cutoffs=[1,5,10,20,50,75,100]):
	metrics = {}
	
	# MRR
	mrrs = np.zeros(len(cutoffs))
	for pred, label in zip(preds, labels):
		jump = False
		for i, x in enumerate(pred, 1):
			if x in label:
				for k, cutoff in enumerate(cutoffs):
					if i <= cutoff:
						mrrs[k] += 1 / i
				jump = True
			if jump:
				break
	mrrs /= len(preds)
	for i, cutoff in enumerate(cutoffs):
		mrr = mrrs[i]
		metrics[f"MRR@{cutoff}"] = mrr

	# Recall
	recalls = np.zeros(len(cutoffs))
	resp_ranks = []
	for pred, label in zip(preds, labels):
		for k, cutoff in enumerate(cutoffs):
			recall = np.intersect1d(label, pred[:cutoff])
			if cutoff == 100: 
				rank = [(pred[:cutoff].index(l) if l in pred[:cutoff] else -1) for l in label]
				resp_ranks.append(rank)
			recalls[k] += len(recall) / len(label)
	recalls /= len(preds)
	for i, cutoff in enumerate(cutoffs):
		recall = recalls[i]
		metrics[f"Recall@{cutoff}"] = recall

	return metrics, resp_ranks



def build_datasets(args):
	test_data = json.load(open(args.test_data)) 
	eval_qs = []
	for q in test_data:
		query, p_ids = q['query'], q['pos']
		eval_qs.append({'query':query, 'positive':p_ids}) #, 'pos_id':pos_ids})	
	return eval_qs




def init_bm25model(args):
	if args.bm25_ver == 'tune':
		return  BM25TnInfer(args)	
	elif args.bm25_ver == 'orig': 
		return BM26Ret(args)
	elif args.bm25_ver == 'mut':
		return BM25Mut(args)





def main(args):
	
	if args.model == 'bm25':
		ranker =  init_bm25model(args)
		eval_data = json.load(open(args.test_data)) #build_datasets(args, idx)
		indices = ranker.search(eval_data, args.k)
	else:	
		ranker = RankerInfer(args)
		corpus = json.load(open(args.corpus_path))
		corpus = [c['txt'] for _,c in corpus.items()]
		faiss_index = dense_index(ranker, corpus, args)
		eval_data = json.load(open(args.test_data))[:256]
		#queries = [s['query'] for s in eval_data]

		_, indices = search(
			model=ranker, 
			queries=eval_data, 
			faiss_index=faiss_index, 
			k=args.k, 
			batch_size=args.batch_size, 
			max_length=args.max_query_length
		)

	#	np.save('saved/m3_ret_results.npy' ,indices)
	retrieval_results = []
	for indice in indices:
		# filter invalid indices
		indice = indice[indice != -1].tolist()
		retrieval_results.append(indice)

	
	ground_truths = []
	for sample in eval_data:
		ground_truths.append(sample["pos"])

	all_metrics = {'rank':{}}
	all_metrics['rank'], resp_ranks = evaluate(retrieval_results, ground_truths)

	print(f'Ranking Eval: {all_metrics}')
	
	
	#for i,r in enumerate(resp_ranks):
	#	eval_data[i]['resp_rank'] = r
	
	#json.dump(eval_data, open('../evals/m3_test_data_ranked_resp.json','w'))
	#json.dump(all_metrics, open(args.output_path,'w'))









if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", default='intfloat/e5-base-v2')
	parser.add_argument("--model_name_or_path", default='/home/ubuntu/projects/regnlp/output/e5-base-rn/ep-2')
	parser.add_argument("--bm25_chkpnt", default='/home/ubuntu/projects/regnlp/output/bm25_idft_rnd_ep_80.pt')

	parser.add_argument('--test_data', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/test.json')
	parser.add_argument('--train_path', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/train.json')
	parser.add_argument('--corpus_path', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/corpus.json')
	parser.add_argument('--index_path', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/index.json')
	parser.add_argument("--output_path", default='../evals/e5_dn')

	parser.add_argument("--max_query_length", default=96)
	parser.add_argument("--batch_size", default=128)
	parser.add_argument('--load_rank_results', type=bool, default=False)
	parser.add_argument('--normlized', type=bool, default=False)
	parser.add_argument("--sp_tokens", default=False)
	parser.add_argument('--do_rerank', type=bool, default=False)

	parser.add_argument("--passage_batch_size", default=64)
	parser.add_argument('--chunked', type=bool, default=False)
	parser.add_argument('--max_chnk_length', type=int, default=500)
	parser.add_argument('--max_chunks_per_passage', type=int, default=1)
	parser.add_argument('--chnk_overlap', type=int, default=50)	

	parser.add_argument("--remove_stop_words", default=True)
	parser.add_argument("--bm25_tok", choices=['split', 'stem', 'gpt4'], default='gpt4')
	parser.add_argument("--k1", default=1.5)
	parser.add_argument("--b", default=.75)
	parser.add_argument("--epsilon", default=.25)
	parser.add_argument("--avgdl_bias", default= 0.)
	parser.add_argument("--didf_c", default=.5)
	parser.add_argument("--qidf_c", default= .5)

	parser.add_argument("--model", choices=['dense','bm25'], default='dense')
	parser.add_argument("--bm25_ver", choices=['tune','mut', 'orig'], default='tune')
	parser.add_argument("--k", default=10)
	parser.add_argument("--index_factory", default='Flat')
	parser.add_argument("--fp16", default=True)
	parser.add_argument("--device", default='cuda:2')

	args = parser.parse_args()
	main(args)
