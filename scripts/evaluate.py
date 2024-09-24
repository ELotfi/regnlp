import faiss
import torch
import logging
import pandas as pd
import json, random, os
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from stop_words import get_stop_words
import tiktoken
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logger = logging.getLogger(__name__)
logging.disable(logging.WARNING)




def convert2format(queries, corpus, use_tokens=False):
	new_q, mapping = {}, {}
	new_c = {k:((use_tokens * 'passage:') + v['txt']) for k,v in corpus.items()}
	for i,q in enumerate(queries):
		new_q[str(i)] = (use_tokens * 'query:') + q['query']
		mapping[str(i)] = [str(p) for p in q['pos']]
	return new_c, new_q, mapping





def main(args):
	corpus = json.load(open(args.corpus_path))
	queries = json.load(open(args.test_data))
	new_c, new_q, mapping = convert2format(queries, corpus)

	model = SentenceTransformer(args.model_name_or_path)
	ir_evaluator = InformationRetrievalEvaluator(
		queries=new_q,
		corpus=new_c,
		relevant_docs=mapping,
		show_progress_bar=True,
		precision_recall_at_k=[1,5,10,20,50,100],
		accuracy_at_k= [1,5,10,20,50,100],
		map_at_k = [1,10,100],
		ndcg_at_k= [1,10,100],
		mrr_at_k = [1,10,100]
	)
	results = ir_evaluator.compute_metrices(model)
	print(results)
	if args.save_results:
		parts = args.model_name_or_path.split('/')
		name = (parts[-1].startswith('ep-') * (parts[-2]+'-')) + parts[-1] 
		json.dump(results, open(f'{args.output_path}/{name}.json','w'))

	#ir_evaluator.output_scores()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name_or_path", default='/home/ubuntu/projects/regnlp/output/e5-base-v2-bm-sp/ep-2')
	parser.add_argument('--test_data', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/test.json')
	parser.add_argument('--corpus_path', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/corpus.json')
	parser.add_argument('--output_path', type=str, default='/home/ubuntu/projects/regnlp/results/dense')
	parser.add_argument("--save_results", default=True)
	parser.add_argument("--use_tokens", default=True)
	args = parser.parse_args()
	main(args)