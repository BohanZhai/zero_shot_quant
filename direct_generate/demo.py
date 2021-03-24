import numpy as np
import torch
from transformer.tokenization import BertTokenizer
from transformer.modeling import BertForMaskedLM
import math
import time
import logging
import sys

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

pretrained_bert_model = f"/rscratch/bohan/ZQBert/zero-shot-qbert/Berts/mrpc_base_l12/"
#pretrained_bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
model = BertForMaskedLM.from_pretrained(pretrained_bert_model)

mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

model.eval()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx
  
  
def get_init_text(seed_text, max_len, batch_size = 1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    #if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))
    
    return tokenize_batch(batch)

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))

def parallel_sequential_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep
    
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=seed_len+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = idxs[jj]
            
        if verbose and np.mod(ii+1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
            print("iter", ii+1, " ".join(for_print))
            
    return untokenize_batch(batch)

def parallel_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True, 
                        cuda=False, print_every=10, verbose=True):
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
            
        if verbose and np.mod(ii, print_every) == 0:
            print("iter", ii+1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))
    
    return untokenize_batch(batch)
            
def sequential_generation(seed_text, batch_size=10, max_len=15, leed_out_len=15, 
                          top_k=0, temperature=None, sample=True, cuda=False):
    """ Generate one word at a time, in L->R order """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_len):
        inp = [sent[:seed_len+ii+leed_out_len]+[sep_id] for sent in batch]
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        idxs = generate_step(out, gen_idx=seed_len+ii, top_k=top_k, temperature=temperature, sample=sample)
        for jj in range(batch_size):
            batch[jj][seed_len+ii] = idxs[jj]
        
    return untokenize_batch(batch)


def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25, 
             generation_mode="parallel-sequential",
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                                   cuda=cuda, verbose=False)
        elif generation_mode == "sequential":
            batch = sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, 
                                          temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                          cuda=cuda)
        elif generation_mode == "parallel":
            batch = parallel_generation(seed_text, batch_size=batch_size,
                                        max_len=max_len, top_k=top_k, temperature=temperature, 
                                        sample=sample, max_iter=max_iter, 
                                        cuda=cuda, verbose=False)
        
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()
        
        sentences += batch
    return sentences

def main():
    n_gpu = torch.cuda.device_count()

    n_samples = 2
    batch_size = 2
    max_len = 40
    top_k = 100
    temperature = 1.0
    generation_mode = "parallel-sequential"
    leed_out_len = 2 # max_len
    burnin = 250
    sample = True
    max_iter = 300

    # Choose the prefix context
    seed_text = "[CLS]".split()
    bert_sents = generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                        generation_mode=generation_mode,
                        sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                        cuda=cuda)

    for sent in bert_sents:
        printer(sent, should_detokenize=True)


if __name__ == "__main__":
    main()
