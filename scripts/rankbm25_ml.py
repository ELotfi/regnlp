import math, json, random, os
import numpy as np
from multiprocessing import Pool, cpu_count
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from stop_words import get_stop_words
import tiktoken
from tqdm import tqdm
import argparse
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from rankbm25 import BM25Okapi
stop_words = get_stop_words('en')
random.seed(42)






class BM25TnData(Dataset):
	def __init__(self, args, split = None, voc = None, idx2tok = None, tok2idx = None, encoded_corpus = None, only_corpus=False) -> None:
		self.args = args
		self.split = split 
		self.tokenizer = tiktoken.encoding_for_model("gpt-4")
		#self.pos_idx = json.load(open(args.index_path))
		self.voc = voc 
		self.idx2tok = idx2tok
		self.tok2idx = tok2idx
		self.encoded_corpus = encoded_corpus
		if voc is None: 
			assert split == 'train'
			self.build_voc()
			self.encode_corpus()
		self.voc_size = len(self.voc)
		print(f'Vocab size: {self.voc_size}')
		if not only_corpus: self.build_dataset(split)


	def tokenize(self, text, is_query=False):
		text = text.lower()
		if self.args.remove_stop_words:
			text = ' '.join([w for w in text.split() if w not in stop_words])
		tokens = self.tokenizer.encode(text)
		if is_query:
			tokens = [t for t in tokens if t in self.voc]
		return tokens
		#return [str(token) for token in tokens]


	def build_voc(self):
		corpus = json.load(open(self.args.corpus_path))
		tok_corpus, voc = [], []
		for _,v in tqdm(corpus.items()):
			toks = self.tokenize(v['txt'])
			tok_corpus.append(toks)
			voc += toks
		self.voc = list(set(voc))
		self.idx2tok = {i:self.voc[i] for i in range(len(self.voc))}
		self.tok2idx = {self.voc[i]:i for i in range(len(self.voc))}
		self.corpus_tok = tok_corpus
	
	
	def encode(self, tokens=None, text=None, is_query=False):
		if tokens is not None:
			#assert all([t in self.voc for t in tokens])
			return [self.tok2idx[t] for t in tokens]
		elif text is not None:
			text_toks = self.tokenize(text, is_query)
			return [self.tok2idx[t] for t in text_toks]
	
	
	def encode_corpus(self):
		self.encoded_corpus = [self.encode(tokens=c) for c in self.corpus_tok]



	def build_dataset(self, split):
		print(f'Building the {split} set ...')
		args = self.args
		paths = {'train':args.train_path, 'dev':args.dev_path} #, 'test':args.test_path}
		queries = json.load(open(paths[split])) #[:500]
		samples = []
		for q in tqdm(queries):
			q_enc = self.encode(text=q['query'], is_query=True)
			pool = [random.choice(q['pos'])] if args.sample_p else q['pos']
			for p in pool:
				samples.append({'q':q_enc, 'p':p})
		random.shuffle(samples)
		self.samples = samples
		print(f'{len(samples)} Samples.')



	def __len__(self):
		return len(self.samples)


	def to_tensor(self, indices):
		out = [0]*self.voc_size
		for i in indices: out[i]+=1
		return torch.tensor(out)



	def __getitem__(self, index):
		sample = self.samples[index]
		pos = self.encoded_corpus[sample['p']]
		return self.to_tensor(sample['q']), self.to_tensor(pos) 


	def collate(self, batch):
		queries, posdocs = zip(*batch)
		queries = torch.stack(queries)
		posdocs = torch.stack(posdocs)
		return queries, posdocs



##########################################################

class BM25TnInfer(BM25Okapi):
	def __init__(self, args):
		self.args = args 
		model = torch.load(args.bm25_chkpnt, weights_only=True)
		self.k1, self.b, self.eps, self.avglb, idf = model['k1'].item(), model['b'].item(), model['epsilon'].item(), model['avgdl_bias'].item(), model['idf'].numpy().tolist()
		data = BM25TnData(args, split='train', only_corpus=True)
		self.voc, self.corpus, self.tok2idx, self.idx2tok = data.voc, data.encoded_corpus, data.tok2idx, data.idx2tok
		assert len(self.voc) == len(idf), 'voc should be the same length as idf'
		self.idf = {i:j for i,j in enumerate(idf)} #self.convert_idf(idf)
		super().__init__(self.corpus, idf=self.idf, k1=self.k1, b=self.b, epsilon=self.eps)
		self.tokenizer = tiktoken.encoding_for_model("gpt-4")


	def tokenize(self, text):
		text = text.lower()
		if self.args.remove_stop_words:
			text = ' '.join([w for w in text.split() if w not in stop_words])
		tokens = self.tokenizer.encode(text)
		tokens = [t for t in tokens if t in self.voc]
		return [self.tok2idx[t] for t in tokens]


	def get_topk(self, query, k):
		query_tok = self.tokenize(query)
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








class BM25Tune(nn.Module):
	def __init__(self, args, corpus, voc_size):
		super().__init__()
		self.args = args
		self.corpus_size = len(corpus)
		self.voc_size = voc_size

		self.k1 = nn.Parameter(torch.tensor(args.k1), requires_grad=True) # torch.tensor(args.k1, requires_grad=True)
		self.b =  nn.Parameter(torch.tensor(args.b), requires_grad=True) #torch.tensor(args.b, requires_grad=True)
		self.epsilon = nn.Parameter(torch.tensor(args.epsilon), requires_grad=True) #torch.tensor(args.epsilon, requires_grad=True)
		self.avgdl_bias = nn.Parameter(torch.tensor(0.), requires_grad=True) #torch.tensor(0., requires_grad=True)
		#self.idf_bias = nn.Parameter(torch.tensor(0.5), requires_grad=True) # torch.tensor(0.5, requires_grad=True)
		#self.idf = nn.Parameter(torch.zeros(voc_size, dtype=torch.float, requires_grad=True))

		self.doc_freqs = []
		self.idf = [0]*voc_size
		self.doc_len = []
		nd = self._initialize(corpus)
		self._calc_idf(nd)


	def _initialize(self, corpus):
		nd = [0]*self.voc_size  # word -> number of documents with word
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
		self.avgdl = (num_doc / self.corpus_size) + self.avgdl_bias
		self.doc_len = torch.tensor(self.doc_len)
		return nd



	def _calc_idf(self, nd):
		if self.args.randinit_idf:
			self.idf = nn.Parameter(torch.ones(self.voc_size)*5., requires_grad=True)
		else: 
			# collect idf sum to calculate an average idf for epsilon value
			idf_sum = 0
			# collect words with negative idf to set them a special epsilon value.
			# idf can be negative if word is contained in more than half of documents
			negative_idfs = []
			for word, freq in enumerate(nd):
				idf = math.log(self.corpus_size - freq + .5) - math.log(freq + .5)
				self.idf[word] = idf
				idf_sum += idf
				if idf < 0:
					negative_idfs.append(word)
			self.average_idf = idf_sum / len(self.idf)

			eps = self.epsilon * self.average_idf
			for word in negative_idfs:
				self.idf[word] = eps
			
			#self.idf = torch.tensor(self.idf)
			self.idf = nn.Parameter(torch.tensor(self.idf), requires_grad=True)



	def forward(self, batch):
		loss_fc = torch.nn.CrossEntropyLoss()
		device = self.args.device
		queries, docs = batch 
		queries, docs = queries.to(device), docs.to(device)
		doc_len = docs.sum(1, keepdim=True).repeat(1,docs.shape[-1]) /self.avgdl
		docs_idf = self.idf.to(device) * ((self.k1 + 1) * docs ) / (docs + self.k1 *(1 - self.b + self.b * doc_len))
		scores = torch.matmul(queries.float(), docs_idf.transpose(1,0))
		#scores = scores.softmax()
		labels = torch.arange(0, scores.shape[-1], dtype=torch.long, device=scores.device)
		loss = loss_fc(scores, labels)
		return loss

		




def prepare_dataloaders(args):
	train_dataset = BM25TnData(args, split='train')
	voc, idx2tok, tok2idx, encoded_corpus = train_dataset.voc, train_dataset.idx2tok, train_dataset.tok2idx, train_dataset.encoded_corpus
	valid_dataset = BM25TnData(args, split='dev', voc=voc, idx2tok=idx2tok, tok2idx=tok2idx, encoded_corpus=encoded_corpus)

	train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle=True, collate_fn=train_dataset.collate, drop_last=True)
	valid_loader = DataLoader(valid_dataset, batch_size = args.valid_batch_size, shuffle=False, collate_fn=valid_dataset.collate, drop_last=True)

	return train_loader, valid_loader, encoded_corpus, voc



def update_logger(logger, parameters, loss , step):
	logger.add_scalar('Train/ Loss', loss , step)
	for n,p in parameters:
		if n!='idf': logger.add_scalar(f'Train/ {n}', p.data.item() , step)




def train_epoch(args, model, train_data, global_step, opt, logger):
	model.train()
	model.zero_grad()
	step_loss = 0
	for i, batch in enumerate(tqdm(train_data)):
		output = model(batch)
		loss = output/args.accumulate_grad
		loss.backward()
		step_loss += loss.item()

		if (i+1)%args.accumulate_grad==0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			opt.step()
			opt.zero_grad()
			update_logger(logger, model.named_parameters(), step_loss , global_step)
			global_step += 1
			step_loss = 0
	return global_step
	


def eval_epoch(model, valid_data, epoch, logger):
	model.eval()
	epoch_loss = []
	for batch in tqdm(valid_data):
		with torch.no_grad():
			loss = model(batch)
		epoch_loss.append(loss.item())
	epoch_loss = np.array(epoch_loss).mean()
	logger.add_scalar('Valid/ Loss', epoch_loss , epoch)




def main(args):
	train_loader, valid_loader, encoded_corpus, voc = prepare_dataloaders(args)
	model = BM25Tune(args, encoded_corpus, len(voc))
	torch.save(model.state_dict(), args.output_path + 'bm25.pt')
	model.to(args.device)
	model.train()
	
	print(len(train_loader), len(valid_loader))
	opt = AdamW(model.parameters(), lr=args.lr)
	logger = SummaryWriter(comment=args.logger_id)

	#s_total = len(train_loader) // args.accumulate_grad * args.epochs
	#schdlr = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=s_total)
	#scaler = GradScaler() if args.fp16 else None

	global_step = 0
	for epoch in range(1, args.epochs + 1):
		global_step = train_epoch(args, model, train_loader, global_step, opt, logger)
		if args.do_eval:
			eval_epoch(model, valid_loader, epoch, logger)
		if args.save_model and epoch%20==0  :
			torch.save(model.state_dict(), args.output_path + f'bm25_idft_rnd_ep_{epoch}.pt')







if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--corpus_path", default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/corpus.json')
	parser.add_argument("--output_path", default='/home/ubuntu/projects/regnlp/output/')
	parser.add_argument('--train_path', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/train.json')
	parser.add_argument('--dev_path', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/dev.json')
	parser.add_argument("--logger_id", default='bm25_rndidf')
	parser.add_argument("--do_eval", default=True)
	parser.add_argument("--save_model", default=True)

	parser.add_argument('--epochs', default= 80, type=int)
	parser.add_argument('--train_batch_size', default= 100, type=int)  # 2 for hard negs
	parser.add_argument('--valid_batch_size', default= 100, type=int)
	parser.add_argument('--accumulate_grad', default= 4, type=int)   # 8 for hard negs
	parser.add_argument('--max_grad_norm', default= 1., type=float)

	parser.add_argument("--remove_stop_words", default=True)
	parser.add_argument("--sample_p", default=True)
	parser.add_argument("--randinit_idf", default=True)
	parser.add_argument("--k1", default=1.5)
	parser.add_argument("--b", default=.75)
	parser.add_argument("--epsilon", default=.25)

	parser.add_argument('--lr', type=float, default=1e-3)	
	parser.add_argument('--seed', default= 42, type=int)
	parser.add_argument("--device", default='cpu')

	args = parser.parse_args()
	#args.device = torch.device(args.device)	
	main(args)








"""

	parser.add_argument('--num_negatives', type=int, default=5) # 5 for hard neg
	parser.add_argument('--neg_interval', default=[2,50])
	parser.add_argument('--negatives', choices=['rand','hard'], default='hard')
	parser.add_argument('--neg_mine_model', choices=['bm25','dense'], default='dense')
	parser.add_argument("--index_factory", default='Flat')
	parser.add_argument("--negs_path", default='../data/')	
	parser.add_argument("--save_negs", default=False)
	parser.add_argument("--load_negs", default=True)

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
"""