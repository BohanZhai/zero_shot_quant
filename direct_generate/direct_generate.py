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
from transformer.modeling import BertForMaskedLM, BertForSequenceClassification

sys.path.append("..")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generate(object):
    def __init__(self, tokenizer, LM_model, generation_mode, batch_size, max_len, temperature, burnin, iter_num, top_k):
        self.tokenizer = tokenizer
        self.model = LM_model
        self.generation_mode = generation_mode
        self.batch_size = batch_size
        self.max_len = max_len
        self.temperature = temperature
        self.burnin = burnin
        self.iter_num = iter_num
        self.top_k = top_k

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
  
    def get_init_text(self, length):
        """ Get initial sentence by padding masks"""
        # batch = [[self.CLS] + [self.MASK] * length + [self.SEP] for _ in range(self.batch_size)]
        batch = [[self.cls_id] + [np.random.randint(1000, 30522) for _ in range(length)] + [self.sep_id] for _ in range(self.batch_size)]
        return batch

    def generate(self):
        length = np.random.randint(1, self.max_len+1)

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

    def parallel_sequential_generation(self, length):
        """ Generate for one random position at a timestep"""
        batch = self.get_init_text(length)
        
        for i in range(self.iter_num):
            position = np.random.randint(0, length)
            for j in range(self.batch_size):
                batch[j][position+1] = self.mask_id
            inp = torch.tensor(batch).to(device)
            out = self.model(inp)
            topk = self.top_k if (i >= self.burnin) else 0
            idxs = self.generate_step(out, gen_idx=position+1, top_k=topk, sample=(i < self.burnin))
            for j in range(self.batch_size):
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
    def __init__(self, tokenizer, TA_model):
        self.tokenizer = tokenizer
        self.model = TA_model

    def generate(self, string_batch):   
        """ Given a list of sentences, call TA_model to generate labels """
        # inputs = self.tokenizer(string_batch, padding = True)
        inputs = torch.LongTensor(string_batch).cuda()
        logits = self.model(inputs)
        # logits = self.model(string_batch)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        outputs = prob.argmax(dim=-1)
        return prob

    def string_generate(self, string_batch):   
        """ Given a list of sentences, call TA_model to generate labels """
        inputs = self.tokenizer(string_batch, padding = True)
        inputs = torch.LongTensor(inputs).cuda()
        logits = self.model(inputs)
        # logits = self.model(string_batch)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        outputs = prob.argmax(dim=-1)
        return outputs

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

    parser.add_argument("--model", default=None, type=str, 
                        help="Fine-tuned Bert model is under is folder")
    parser.add_argument("--output_dir", default=None, type=str, 
                        help="Generated output file will be placed under this folder")
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
    parser.add_argument("--batch_num", default=10, type=int, 
                        help="How many batches to generate")
    parser.add_argument("--batch_size", default=100, type=int,
                        help="How many sentence to generate for one batch")
    parser.add_argument("--top_k", default=100, type=int,
                        help="Choose from top k words instead of full distribution")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--burnin", default=10, type=int,
                        help="For non-sequential generation, for the first burnin steps, sample from the entire next word distribution, instead of top_k")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed to reproduce results")


    args = parser.parse_args()
    # logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # TO DO: set fine-tuned model for now, later take from args.model
    pretrained_bert_model_gen = f"/rscratch/bohan/ZQBert/zero-shot-qbert/Berts/gen_base_l12/"
    pretrained_bert_model_sst = f"/rscratch/bohan/ZQBert/zero-shot-qbert/Berts/sst_base_l12/"
    data_dir = f"/rscratch/bohan/ZQBert/GLUE-baselines/glue_data/SST-2"

    # Prepare two models for generation and labeling
    tokenizer_gen = BertTokenizer.from_pretrained(pretrained_bert_model_gen)
    tokenizer_sst = BertTokenizer.from_pretrained(pretrained_bert_model_sst)
    LM_model = BertForMaskedLM.from_pretrained(pretrained_bert_model_sst)
    LM_model.eval()
    LM_model.to(device)

    TA_model = BertForSequenceClassification.from_pretrained(pretrained_bert_model_sst)
    TA_model.eval()
    TA_model.to(device)

    # task_name = args.task_name.lower()
    if args.max_len > args.iter_num:
        logger.info("Iteration less than length of a sentence will possibly result in a sentence with [MASK]")
    
    with torch.no_grad():
        if args.generate_label:
            get_label = Get_label()
            labeler = Label(tokenizer_sst, TA_model)
            batch = []
            line_num = args.batch_size
            for line in get_label.get_train_examples(data_dir):
                batch.append(line[-1])
                line_num -= 1
                if line_num <= 0:
                    prob_tensor = labeler.string_generate(batch)
                    line_num = args.batch_size
                    with open('./output_labels.tsv', 'a+') as out_file:
                        tsv_writer = csv.writer(out_file, delimiter='\t')
                        for i in range(len(prob_tensor)):
                            sentence = str(string_batch[i])
                            # label = str(labels[i].cpu())
                            prob = str(prob_tensor[i].cpu()[0].item())
                            # TODO: replace this line
                            tsv_writer.writerow([sentence, prob])
                    line_num = args.batch_size
                    batch = []
        else:
            generator = Generate(tokenizer_sst, LM_model, args.generation_mode, args.batch_size, args.max_len, 
                                    args.temperature, args.burnin, args.iter_num, args.top_k)
            labeler = Label(tokenizer_sst, TA_model)

            for batch in range(args.batch_num):
                #sentence_batch = generator.generate()
                length = np.random.randint(1, args.max_len+1)
                sentence_batch = generator.get_init_text(length)

                prob_tensor = labeler.generate(sentence_batch)
                # s = "are more deeply thought through than in most ` right-thinking ' films"
                # t = torch.unsqueeze(torch.tensor(tokenizer.encode(s)), 0).to(device)
                # new_label = labeler.generate(t)


                with open('./output_labels.tsv', 'a+') as out_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    for i in range(len(prob_tensor)):
                        sentence = str(sentence_batch[i])
                        # label = str(labels[i].cpu())
                        prob = str(prob_tensor[i].cpu()[0].item())
                        # TODO: replace this line
                        tsv_writer.writerow([sentence, prob])


            logger.info("Having been generating {} batches".format(str(batch+1)))


if __name__ == "__main__":
    main()

