"""
Generate sentence and label directly from fine-tuned Bert.
Idea from "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model" (https://arxiv.org/pdf/1902.04094.pdf).
"""

import argparse
import logging
import sys
import csv
import torch
import os
import numpy as np

from transformers import BertTokenizer
from transformers import BertForMaskedLM, BertForSequenceClassification
# from tinybert_aug import * 

sys.path.append("..")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generate(object):
    def __init__(self, tokenizer, LM_model, generation_mode, batch_size, max_len, temperature, burnin, iter_num, top_k, task):
        self.tokenizer = tokenizer
        self.model = LM_model
        self.generation_mode = generation_mode
        self.batch_size = batch_size
        self.max_len = max_len
        self.temperature = temperature
        self.burnin = burnin
        self.iter_num = iter_num
        self.top_k = top_k
        self.task = task

        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.MASK = '[MASK]'
        self.mask_id = tokenizer.convert_tokens_to_ids([self.MASK])[0]
        self.sep_id = tokenizer.convert_tokens_to_ids([self.SEP])[0]
        self.cls_id = tokenizer.convert_tokens_to_ids([self.CLS])[0]
    
    def tokenize_batch(self, batch):
        return [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]

    def untokenize_batch(self, batch):
        return [self.tokenizer.convert_ids_to_tokens(sent) for sent in batch]

    def sentenize_batch(self, batch):
        return [self.tokenizer.convert_tokens_to_string(sent) for sent in batch]

    def generate_step(self, out, gen_idx, top_k=0, sample=False):
        """ Generate a word from from out[gen_idx]
        
        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k 
        """
        logits = out[:, gen_idx]
        if self.temperature:
            logits = logits / self.temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if self.batch_size!=1 else idx
  
    def get_init_text(self, length, task):
        """ Get initial sentence by padding masks"""
        # batch = [[self.CLS] + [self.MASK] * length + [self.SEP] for _ in range(self.batch_size)]
        if task == 'sst':
            batch = [[self.cls_id] + [np.random.randint(1000, 30522) for _ in range(length)] + [self.sep_id] for _ in range(self.batch_size)]
        elif task == 'mrpc' or task == 'rte' or task == 'qnli' or task == 'mnli':
            batch = [[self.cls_id] + [np.random.randint(1000, 30522) for _ in range(length)] + [self.sep_id] \
            + [np.random.randint(1000, 30522) for _ in range(length)] + [self.sep_id] for _ in range(self.batch_size)]
        else:
            raise NotImplementedError()
        return batch

    def generate(self):
        length = np.random.randint(2, self.max_len+1)
        if self.generation_mode == "parallel-sequential":
            sentences = self.parallel_sequential_generation(length)

        # not implemented
        elif generation_mode == "sequential":
            raise NotImplementedError()
            batch = sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, 
                                        temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                        cuda=cuda)
        elif generation_mode == "parallel":
            raise NotImplementedError()
            batch = parallel_generation(seed_text, batch_size=batch_size,
                                        max_len=max_len, top_k=top_k, temperature=temperature, 
                                        sample=sample, max_iter=max_iter, 
                                        cuda=cuda, verbose=False)
        else:
            raise ValueError("Generation mode not found: %s" % self.generation_mode)

        return sentences

    def parallel_sequential_generation(self, length, previous_batch=None, iter_num=None, batch_size=None):
        """ Generate for one random position at a timestep"""
        if previous_batch:
            batch = previous_batch
        else:
            batch = self.get_init_text(length, task=self.task)

        if not iter_num:
            iter_num = self.iter_num

        if not batch_size:
            batch_size = self.batch_size

        for i in range(iter_num):
            position = np.random.randint(0, length)
            for j in range(batch_size):
                if (batch[j][position+1] != self.sep_id) and (batch[j][position+1] != self.cls_id):
                    batch[j][position+1] = self.mask_id

            inp = torch.tensor(batch).to(device)
            out = self.model(inp)['logits']
            topk = self.top_k if (i >= self.burnin) else 0
            idxs = self.generate_step(out, gen_idx=position+1, top_k=topk, sample=(i < self.burnin))
            for j in range(batch_size):
                if isinstance(idxs, int):
                    batch[0][position+1] = idxs
                else:
                    batch[j][position+1] = idxs[j]

        return batch


    def parallel_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True):
        """ Generate for all positions at each time step """
        seed_len = len(seed_text)
        batch = get_init_text(seed_text, max_len, batch_size)
        
        for ii in range(max_iter):
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            out = model(inp)
            for kk in range(max_len):
                idxs = generate_step(out, gen_idx=seed_len+kk, top_k=top_k, temperature=temperature, sample=sample)
                for jj in range(batch_size):
                    batch[jj][seed_len+kk] = idxs[jj]
        print("String", self.untokenize_batch(batch))
        return batch

    def process(self, sentences):
        """ Finish building a batch by breaking at CLS """
        new_batch = []
        for s in sentences:
            new_batch.append(" ".join(s))
        return new_batch

class Label(object):
    def __init__(self, tokenizer, TA_model, task='sst'):
        self.tokenizer = tokenizer
        self.model = TA_model
        self.task = task

    def generate(self, string_batch):   
        """ Given a list of sentences, call TA_model to generate labels """
        # inputs = self.tokenizer(string_batch, padding = True)
        inputs = torch.LongTensor(string_batch).cuda()
        logits = self.model(inputs)['logits']
        # logits = self.model(string_batch)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        # outputs = prob.argmax(dim=-1)
        return prob

    def string_generate(self, string_batch):   
        """ Given a list of sentences, call TA_model to generate labels """
        inputs = self.tokenizer(string_batch, padding = True)
        inputs = torch.LongTensor(inputs['input_ids']).cuda()
        logits = self.model(inputs)
        # logits = self.model(string_batch)
        prob = torch.nn.functional.softmax(logits['logits'], dim=-1)
        # outputs = prob.argmax(dim=-1)
        return prob, inputs.tolist()



class Get_label(object):
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--LM_model", default=None, type=str, 
                        help="Fine-tuned Bert with LM model is under is folder")
    parser.add_argument("--TA_model", default=None, type=str, 
                        help="Fine-tuned Bert model is under is folder")
    parser.add_argument("--output_dir", default=None, type=str, 
                        help="Generated output file will be placed under this folder")
    parser.add_argument("--file_name", default=None, type=str, 
                        help="Generated output file will be named according to this")
    parser.add_argument("--generation_mode", default="parallel-sequential", type=str, 
                        help="Mode of generating sentence")
    parser.add_argument("--generate_label", default=False, type=bool,
                        help="Whether get label for baseline")
    # parser.add_argument("--task_name", default=None, type=str, 
    #                     help="Which head to use") # sequence classification for now
    parser.add_argument("--max_len", default=50, type=int, 
                        help="Max sequence length to generate")
    parser.add_argument("--iter_num", default=100, type=int, 
                        help="Iteration of repeating masking for each sentence")
    parser.add_argument("--batch_num", default=87, type=int, 
                        help="How many batches to generate")
    parser.add_argument("--batch_size", default=100, type=int,
                        help="How many sentence to generate for one batch")
    parser.add_argument("--random", default=False, type=bool,
                        help="Whether to randomly generate data")
    parser.add_argument("--top_k", default=100, type=int,
                        help="Choose from top k words instead of full distribution")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--burnin", default=10, type=int,
                        help="For non-sequential generation, for the first burnin steps, sample from the entire next word distribution, instead of top_k")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed to reproduce results")
    parser.add_argument("--task", default='mrpc', type=str, 
                        help="Input tasks in cola, mnli, qnli, qqp, sts-b, rte, sst-2, wnli, mrpc")


    args = parser.parse_args()
    # logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    data_dir = f"/rscratch/bohan/ZQBert/GLUE-baselines/glue_data/RTE"

    # Prepare two models for generation and labeling
    tokenizer = BertTokenizer.from_pretrained(args.LM_model)
    LM_model = BertForMaskedLM.from_pretrained(args.LM_model)
    LM_model.eval()
    LM_model.to(device)

    TA_model = BertForSequenceClassification.from_pretrained(args.TA_model)
    TA_model.eval()
    TA_model.to(device)

    # task_name = args.task_name.lower()
    if args.max_len > args.iter_num:
        logger.info("Iteration less than length of a sentence will possibly result in a sentence with [MASK]")
    
    with torch.no_grad():
        if args.generate_label:
            get_label = Get_label()
            labeler = Label(tokenizer, TA_model)
            batch = []
            line_num = args.batch_size
            for line in get_label.get_train_examples(data_dir):
                if args.task == 'sst':
                    batch.append(line[-1])
                elif args.task == 'mnli':
                    batch.append(line[1] + line[2])
                elif args.task == 'rte':
                    batch.append(line[8] + line[9])
                line_num -= 1
                if line_num <= 0:
                    prob_tensor, sentences = labeler.string_generate(batch)
                    line_num = args.batch_size
                    with open('./'+args.output_dir+args.file_name+'.tsv', 'a+') as out_file:
                        tsv_writer = csv.writer(out_file, delimiter='\t')
                        for i in range(len(prob_tensor)):
                            sentence = str(sentences[i])
                            prob = str(prob_tensor[i].cpu()[0].item())

                            probs = [prob_tensor[i].cpu()[j].item() for j in range(len(prob_tensor[i].cpu()))]
                            probs = tuple(probs)
                            probs = str(probs)
                            # TODO: replace this line
                            if args.task == 'rte':
                                tsv_writer.writerow([i, sentence, prob])
                            elif args.task == 'mnli':
                                tsv_writer.writerow([i, sentence, probs])
                            else:
                                tsv_writer.writerow([sentence, prob])
                    line_num = args.batch_size
                    batch = []
        else:
            generator = Generate(tokenizer, LM_model, args.generation_mode, args.batch_size, args.max_len, 
                                    args.temperature, args.burnin, args.iter_num, args.top_k, args.task)
            labeler = Label(tokenizer, TA_model)

            for batch in range(args.batch_num):
                #sentence_batch = generator.generate()
                length = np.random.randint(1, args.max_len+1)
                if args.random:
                    sentence_batch = generator.get_init_text(length, args.task)
                else:
                    sentence_batch = generator.generate()


                if args.task == 'mrpc' or args.task == 'rte' or args.task == 'qnli':
                    for i in range(args.batch_size):
                        if np.random.rand() >= 0.5:
                            split_index = sentence_batch[i].index(102)+1
                            sentence_a = sentence_batch[i][:split_index]
                            sentence_batch[i] = sentence_a + sentence_a[1:]
                            sentence_batch[i] = generator.parallel_sequential_generation(len(sentence_batch[i])-1, previous_batch=[sentence_batch[i]], iter_num=5, batch_size=1)[0]
                elif args.task == 'mnli':
                    for i in range(args.batch_size):
                        prob = np.random.rand()
                        if prob < 0.33:
                            split_index = sentence_batch[i].index(102)+1
                            sentence_a = sentence_batch[i][:split_index]
                            sentence_batch[i] = sentence_a + sentence_a[1:]
                            sentence_batch[i] = generator.parallel_sequential_generation(len(sentence_batch[i])-1, previous_batch=[sentence_batch[i]], iter_num=5, batch_size=1)[0]
                        elif 0.33 <= prob < 0.66:
                            split_index = sentence_batch[i].index(102)+1
                            sentence_a = sentence_batch[i][:split_index]
                            random_position = min(max(1, int(np.random.normal(len(sentence_a)//2))), len(sentence_a)-1)
                            temp = sentence_a + sentence_a[1:random_position] + [102]
                            num_pad = len(sentence_a)*2-1 - len(temp)
                            temp += [0] * num_pad
                            sentence_batch[i] = temp

                        
                prob_tensor = labeler.generate(sentence_batch)

                with open('./'+args.output_dir+args.file_name+'.tsv', 'a+') as out_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    for i in range(len(prob_tensor)):
                        sentence = str(sentence_batch[i])
                        # label = str(labels[i].cpu())
                        prob = str(prob_tensor[i].cpu()[0].item())
                        
                        # print('probs')
                        probs = [prob_tensor[i].cpu()[j].item() for j in range(len(prob_tensor[i].cpu()))]
                        probs = tuple(probs)
                        probs = str(probs)
                        # probs = str((prob_tensor[i].cpu().item()[0], \
                        # prob_tensor[i].cpu().item()[1], prob_tensor[i].cpu().item()[2]))

                        if args.task == 'sst':
                            tsv_writer.writerow([sentence, prob])
                        elif args.task == 'mrpc':
                            tsv_writer.writerow([prob, 0, 0, sentence])
                        elif args.task == 'rte' or args.task == 'qnli':
                            tsv_writer.writerow([i, sentence, prob])
                        elif args.task == 'mnli':
                            tsv_writer.writerow([i, sentence, probs])
                        


                logger.info("Having been generating {} batches".format(str(batch+1)))


if __name__ == "__main__":
    main()

