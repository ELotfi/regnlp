from tqdm import tqdm
import torch
import argparse
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel ,set_seed,  get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json, os, random
from torch.utils.tensorboard import SummaryWriter

from bm25gpt import BM25



class ObliQAData(Dataset):
	def __init__(self, args, is_train=True):
		self.args = args
		self.is_train = is_train
		self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
		if args.sp_tokens: self.add_sp_tokens()
		self.corpus = json.load(open(args.corpus_path))
		self.data = self.build_data()



	def __len__(self):
		return len(self.data)


	def add_sp_tokens(self):
		self.tokenizer.add_special_tokens({'additional_special_tokens':['<query>', '<document>']})


	def add_negatives(self, queries):
		if self.args.negatives == 'rand':
			for i in range(len(queries)): queries[i]['neg'] = []
			return queries
		print('Adding hard negatives ...')
		num_neg, (neg_st, neg_en) = self.args.num_negatives, self.args.neg_interval

		if self.args.load_negs:
			file_name = f'{args.negs_path}{"train" if self.is_train else "dev"}_negs.npy'
			top100 = np.load(open(file_name,'rb'))
		else:
			ranker = BM25(args)
			qs = [q['query'] for q in queries]
			top100 = ranker.search(qs)
			if self.args.save_negs:
				file_name = f'{args.negs_path}{"train" if self.is_train else "dev"}_negs.npy'
				np.save(open(file_name,'wb'), top100)
		
		for i in range(len(queries)):
			negs = top100[i][neg_st:neg_en]
			negs = [n for n in negs if n not in queries[i]['pos']]
			negs = negs[:num_neg] if self.args.neg_pick == 'top' else random.sample(negs, num_neg)
			queries[i]['neg'] = negs
		
		return queries


	def build_data(self):
		args = self.args
		queries = json.load(open(args.train_data if self.is_train else args.valid_data))[:args.max_samples]
		data = []
		print('Building dataset...')
		queries = self.add_negatives(queries)
		for i,sample in enumerate(tqdm(queries)):
			data += [{'q':sample['query'], 'p':p, 'n':sample['neg']} for p in sample['pos']]
		return data
	

	
	def __getitem__(self, index):
		sample =  self.data[index]
		query , pos, neg = sample['q'], [sample['p']], sample.get('n', [])
		passages = [self.corpus[str(idx)]['txt'] for idx in (pos + neg)]
		#labels = [1]*len(poss) + [0]*len(negs)
		if self.args.sp_tokens:
			query = f'query: {query}'
			passages = [f'passage: {p}' for p in passages]	
		return query, passages


	def collate(self, batch):
		args = self.args
		queries, passages = zip(*batch)
		passages = [p for ps in passages for p in ps]
		#print(passages)
		queries = self.tokenizer(queries, max_length = args.max_query_length, truncation=True, padding=True, return_tensors='pt')
		passages = self.tokenizer(passages, max_length = args.max_passage_length, truncation=True, padding=True, return_tensors='pt')
		return queries, passages

################################################################

class Ranker(torch.nn.Module):
	def __init__(self, args, tokenizer=None):
		super().__init__()
		self.args = args
		self.model = AutoModel.from_pretrained(args.model_name_or_path)
		self.tokenizer = tokenizer
		self.loss_fc = torch.nn.CrossEntropyLoss()
		if args.sp_tokens: self.model.resize_token_embeddings(len(self.tokenizer))
	
	
	def encode(self, inputs):
		encoded = self.model(**inputs).pooler_output
		if self.args.normlized:
			encoded = torch.nn.functional.normalize(encoded, dim=-1)
		return encoded


	def forward(self, batch):
		args, device = self.args, self.args.device
		queries, passages = batch
		queries = {k:queries[k].to(device) for k in queries}
		passages = {k:passages[k].to(device) for k in passages}

		q_reps = self.encode(queries)
		p_reps = self.encode(passages)
		b, d = q_reps.shape
		if args.negatives == 'rand':
			scores = torch.matmul(q_reps, p_reps.transpose(0,1))/args.temperature
			labels = torch.arange(0, scores.shape[-1] ,dtype=torch.long, device=scores.device)
		else: 
			scores = torch.bmm(q_reps.unsqueeze(1), p_reps.reshape(b, -1, d).transpose(1,2)).squeeze()/self.args.temperature
			labels = torch.zeros((b,), dtype=torch.long, device=scores.device)
		loss = self.loss_fc(scores, labels)
		return loss


	def save_model(self, epoch):
		path = self.args.output_path + f'ep-{epoch}'
		os.makedirs(path, exist_ok=True)
		print('Saving checkpoint ...')
		self.model.save_pretrained(path)
		self.tokenizer.save_pretrained(path)

######################################

class RankerInfer(Ranker):
	def __init__(self, args):
		super().__init__(args)
		self.model.to(args.device)
		self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
		


	@torch.no_grad()
	def encode_corpus(self, corpus):
		print('Encoding Corpus ...')
		passages = self.tokenizer(corpus, max_length = self.args.max_chnk_length, truncation=True, padding=True, return_tensors='pt')
		enc_batches = []
		n, bs = len(corpus), self.args.passage_batch_size
		for i in tqdm(range(0, n, bs)):
			batch = {k:passages[k][i: i + bs].to(self.args.device) for k in ['input_ids', 'attention_mask']}
			enc_batch = super().encode(batch)
			enc_batches.append(enc_batch)
		
		enc_batches = torch.cat(enc_batches, dim=0).cpu().numpy()
		return enc_batches

	
	@torch.no_grad()
	def encode_queries(self, queries, batch_size, max_length):
		tokenized = self.tokenizer(queries, max_length = max_length, truncation=True, padding=True, return_tensors='pt')
		tokenized = {k:v.to(self.args.device) for k,v in tokenized.items()}
		output = []
		print('Encoding Queries ...')
		for i in tqdm(range(0, tokenized['input_ids'].shape[0], batch_size)):
			batch = {k:v[i:i+batch_size, ] for k,v in tokenized.items()}
			encoded = super().encode(batch)
			output.append(encoded)
		return torch.cat(output, dim=0).cpu().numpy()



###################################################################

def prepare_dataloaders(args):
	train_dataset, valid_dataset = ObliQAData(args) , ObliQAData(args, is_train=False)
	train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle=True, collate_fn=train_dataset.collate, drop_last=True)
	valid_loader = DataLoader(valid_dataset, batch_size = args.valid_batch_size, shuffle=False, collate_fn=valid_dataset.collate, drop_last=True)
	tokenizer = train_dataset.tokenizer
	return train_loader, valid_loader, tokenizer


def compute_similarity(q, p):
	return torch.matmul(q, p.transpose(0, 1))


def train_epoch(args, model, train_data, global_step, schdlr, opt, scaler, logger):
	model.train()
	model.zero_grad()
	step_loss = 0
	for i, batch in enumerate(tqdm(train_data)):
		if args.fp16:
			with autocast(): output = model(batch)
		else: output = model(batch)
		loss = output/args.accumulate_grad 
		if args.fp16: scaler.scale(loss).backward()
		else: loss.backward()
		step_loss += loss.item()

		if (i+1)%args.accumulate_grad==0:
			if args.fp16: scaler.unscale_(opt)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			if args.fp16:
				scaler.step(opt)
				scaler.update()
			else: opt.step()
			opt.zero_grad()
			schdlr.step()
			logger.add_scalar('Train/ Loss', step_loss , global_step)
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
	set_seed(args.seed)
	train_loader, valid_loader, tokenizer = prepare_dataloaders(args)
	model = Ranker(args, tokenizer)
	model.to(args.device)
	
	print(len(train_loader), len(valid_loader))
	opt = AdamW(model.parameters(), lr=args.lr)

	s_total = len(train_loader) // args.accumulate_grad * args.epochs
	schdlr = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=s_total)
	scaler = GradScaler() if args.fp16 else None

	global_step = 0
	logger = SummaryWriter(comment=args.logger_id)
	for epoch in range(1, args.epochs + 1):
		global_step = train_epoch(args, model, train_loader, global_step, schdlr, opt, scaler, logger)
		if args.do_eval:
			eval_epoch(model, valid_loader, epoch, logger)
		if args.save_model:
			model.save_model(epoch)






if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name_or_path", default='intfloat/e5-base-v2')
	parser.add_argument("--corpus_path", default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/corpus.json')
	parser.add_argument("--output_path", default='../output/e5-base-v2')
	parser.add_argument('--train_data', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/train.json')
	parser.add_argument('--valid_data', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/dev.json')
	parser.add_argument('--test_data', type=str, default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/test.json')
	parser.add_argument("--logger_id", default='e5-base-obliqa')
	parser.add_argument("--save_model", default=True)
	parser.add_argument("--do_eval", default=True)

	parser.add_argument('--max_samples', type=int, default=100000)
	parser.add_argument('--max_query_length', type=int, default=96)
	parser.add_argument('--max_passage_length', type=int, default=450)  # 500 for hard neg

	parser.add_argument('--num_negatives', type=int, default=10) # 5 for hard neg
	parser.add_argument('--neg_interval', default=[2,20])
	parser.add_argument('--negatives', choices=['rand','hard'], default='hard')
	parser.add_argument('--neg_pick', choices=['top','rnd'], default='rnd')
	parser.add_argument('--neg_mine_model', choices=['bm25','dense'], default='bm25')
	parser.add_argument("--remove_stop_words", default=True)
	parser.add_argument("--index_factory", default='Flat')
	parser.add_argument("--negs_path", default='/home/ubuntu/projects/regnlp/data/ObliQADataset/new/')	
	parser.add_argument("--save_negs", default=False)
	parser.add_argument("--load_negs", default=True)

	parser.add_argument("--sp_tokens", default=True)
	parser.add_argument('--normlized', type=bool, default=False)
	parser.add_argument('--epochs', default= 4, type=int)
	parser.add_argument('--warmup_steps', default= 100, type=int)
	parser.add_argument('--train_batch_size', default= 1, type=int)  # 2 for hard negs
	parser.add_argument('--valid_batch_size', default= 32, type=int)
	parser.add_argument('--accumulate_grad', default=32, type=int)   # 8 for hard negs
	parser.add_argument('--max_grad_norm', default= 1., type=float)
	parser.add_argument('--temperature', default= 1., type=float)
	parser.add_argument("--fp16", default=True)
	parser.add_argument('--lr', type=float, default=2e-5)	
	parser.add_argument('--seed', default= 42, type=int)
	parser.add_argument("--device", default='cuda:1')

	args = parser.parse_args()
	args.device = torch.device(args.device)	

	args.logger_id += ('-rn' if args.negatives=='rand' else '-bm' if args.neg_mine_model=='bm25'  else '-dn') + ('-sp' if args.sp_tokens else '')
	args.output_path += ('-rn' if args.negatives=='rand' else '-bm' if args.neg_mine_model=='bm25'  else '-dn')+ ('-sp' if args.sp_tokens else '') + '/'
	main(args)
