"""
Generate sentence and label directly from fine-tuned Bert.
Idea from "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model" (https://arxiv.org/pdf/1902.04094.pdf).
"""

import argparse
import logging
import sys
import csv
import torch
import numpy as np

from transformers import BertTokenizer
from transformer.modeling import BertForMaskedLM, BertForSequenceClassification

sys.path.append("..")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def _read_tsv(input_file, quotechar=None):
#     """Reads a tab separated value file."""
#     with open(input_file, "r", encoding="utf-8") as f:
#         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#         lines = []
#         for line in reader:
#             if sys.version_info[0] == 2:
#                 line = list(unicode(cell, 'utf-8') for cell in line)
#             lines.append(line)
#         return lines


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#     """Truncates a sequence pair in place to the maximum length."""
#     while True:
#         total_length = len(tokens_a) + len(tokens_b)
#         if total_length <= max_length:
#             break
#         if len(tokens_a) > len(tokens_b):
#             tokens_a.pop()
#         else:
#             tokens_b.pop()

# class DataAugmentor(object):
#     def __init__(self, model, tokenizer, teacher_model, teacher_model_tokenizer, emb_norm, vocab, ids_to_tokens, 
#         M, N, p, p_dist=0.5, pair=False):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.emb_norm = emb_norm
#         self.vocab = vocab
#         self.ids_to_tokens = ids_to_tokens
#         self.M = M
#         self.N = N
#         self.p = p
#         self.p_dist = p_dist
#         self.teacher_model = teacher_model
#         self.teacher_model_tokenizer = teacher_model_tokenizer
#         self.num_labels = teacher_model.num_labels
#         self.label_count = [0 for _ in range(self.num_labels)]
#         self.pair = pair

#     def _word_distance(self, word):
#         p_dist = random.random() < self.p_dist
#         if word not in self.vocab.keys():
#             return []
#         word_idx = self.vocab[word]
#         word_emb = self.emb_norm[word_idx]

#         dist = np.dot(self.emb_norm, word_emb.T)
#         dist[word_idx] = -np.Inf

#         if p_dist:
#             candidate_ids = np.argsort(-dist)[:self.M]
#         else:
#             candidate_ids = np.argsort(dist)[:self.M]

#         return [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

#     def _check_label(self, sent, candidate_sent):
#         tokenized_text = self.teacher_model_tokenizer.tokenize(candidate_sent)
        

#         if self.pair:
#             tokenized_text_a = self.teacher_model_tokenizer.tokenize(sent)
#             _truncate_seq_pair(tokenized_text_a, tokenized_text, max_seq_length - 3, )
#             # tokenized_text = tokenized_text_a + ['[SEP]'] + tokenized_text + ['[SEP]']
#         else:
#             tokenized_text_a = None
#             if len(tokenized_text) > max_seq_length - 2:
#                 tokenized_text = tokenized_text[:(max_seq_length - 2)]
#             # tokenized_text = tokenized_text + ['[SEP]']

#         tokens = ["[CLS]"] + tokenized_text + ["[SEP]"]
#         segment_ids = [0] * len(tokens)

#         if tokenized_text_a:
#             tokens += tokenized_text_a + ["[SEP]"]
#             segment_ids += [1] * (len(tokenized_text_a) + 1)

#         input_ids = self.teacher_model_tokenizer.convert_tokens_to_ids(tokens)
#         input_mask = [1] * len(input_ids)
#         seq_length = len(input_ids)

#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding

        
#         segments_tensor = torch.tensor([segment_ids]).to(device)
#         tokens_tensor = torch.tensor([input_ids]).to(device)
#         input_mask_tensor = torch.tensor([input_mask]).to(device)

#         self.teacher_model.to(device)
#         logits, _, _ = self.teacher_model(tokens_tensor, segments_tensor, input_mask_tensor, is_student=True)
#         label = np.argmax( logits.cpu().numpy()[0] )
#         return label

#     def _masked_language_model(self, sent, word_pieces, mask_id):
#         tokenized_text = self.tokenizer.tokenize(sent)
#         tokenized_text = ['[CLS]'] + tokenized_text
#         tokenized_len = len(tokenized_text)

#         tokenized_text = word_pieces + ['[SEP]'] + tokenized_text[1:] + ['[SEP]']

#         if len(tokenized_text) > 512:
#             tokenized_text = tokenized_text[:512]

#         token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
#         segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) - tokenized_len - 1)

#         tokens_tensor = torch.tensor([token_ids]).to(device)
#         segments_tensor = torch.tensor([segments_ids]).to(device)

#         self.model.to(device)

#         predictions = self.model(tokens_tensor, segments_tensor)

#         word_candidates = torch.argsort(predictions[0, mask_id], descending=True)[:self.M].tolist()
#         word_candidates = self.tokenizer.convert_ids_to_tokens(word_candidates)

#         return list(filter(lambda x: x.find("##"), word_candidates))

#     def _word_augment(self, sentence, mask_token_idx, mask_token):
#         word_pieces = self.tokenizer.tokenize(sentence)
#         word_pieces = ['[CLS]'] + word_pieces
#         tokenized_len = len(word_pieces)

#         token_idx = -1
#         for i in range(1, tokenized_len):
#             if "##" not in word_pieces[i]:
#                 token_idx = token_idx + 1
#                 if token_idx < mask_token_idx:
#                     word_piece_ids = []
#                 elif token_idx == mask_token_idx:
#                     word_piece_ids = [i]
#                 else:
#                     break
#             else:
#                 word_piece_ids.append(i)
#         if mask_token == '[UNK]':
#             print(word_piece_ids, len(word_piece_ids))
#             exit()

#         if len(word_piece_ids) == 1:
#             word_pieces[word_piece_ids[0]] = '[MASK]'
#             candidate_words = self._masked_language_model(
#                 sentence, word_pieces, word_piece_ids[0])
#         elif len(word_piece_ids) > 1:
#             candidate_words = self._word_distance(mask_token)
#         else:
#             logger.info("invalid input sentence!")
        
#         if len(candidate_words)==0:
#             candidate_words.append(mask_token)

#         return candidate_words

#     def augment(self, sent):
#         # candidate_sents = [sent]

#         candidate_sents = []

#         tokens = self.tokenizer.basic_tokenizer.tokenize(sent)
#         candidate_words = {}
#         for (idx, word) in enumerate(tokens):
#             if _is_valid(word) and word not in StopWordsList:
#                 candidate_words[idx] = self._word_augment(sent, idx, word)
#         # logger.info(candidate_words)
#         cnt = 0
#         while cnt < self.N:
#             new_sent = list(tokens)
#             for idx in candidate_words.keys():
#                 candidate_word = random.choice(candidate_words[idx])

#                 x = random.random()
#                 if x < self.p:
#                     new_sent[idx] = candidate_word

#             if " ".join(new_sent) not in candidate_sents:
#                 label = self._check_label( sent, " ".join(new_sent),  )

#                 candidate_sents.append( (' '.join(new_sent), label) )

#                 self.label_count[label] += 1

#             cnt += 1

#         return candidate_sents

# augment_ids = {'MRPC': [3, 4], 'MNLI': [8, 9], 'CoLA': [3], 'SST-2': [0],
#                 'STS-B': [7, 8], 'QQP': [3, 4], 'QNLI': [1, 2], 'RTE': [1, 2]}

# label_ids = {'MRPC': 0, 'MNLI': -1, 'CoLA': 1, 'SST-2': 1,
#             'STS-B': -1, 'QQP': 5, 'QNLI': -1, 'RTE': -1}

# class AugmentProcessor(object):
#     def __init__(self, augmentor, glue_dir, task_name, source="MNLI", label_list=None):
#         self.augmentor = augmentor
#         self.glue_dir = glue_dir
#         self.task_name = task_name

#         self.source = source
#         self.augment_ids = {'MRPC': [3, 4], 'MNLI': [8, 9], 'CoLA': [3], 'SST-2': [0],
#                             'STS-B': [7, 8], 'QQP': [3, 4], 'QNLI': [1, 2], 'RTE': [1, 2]}

#         self.filter_flags = { 'MRPC': True, 'MNLI': True, 'CoLA': False, 'SST-2': True,
#                               'STS-B': True, 'QQP': True, 'QNLI': True, 'RTE': True}
#         self.n_sample = 10000
#         self.label_list = label_list

#         assert self.task_name in self.augment_ids

#     def _process_source(self):
#         source_sents = []
#         if self.source in self.augment_ids:
#             source_dir = os.path.join(self.glue_dir, self.source)
#             source_train_samples = _read_tsv(os.path.join(source_dir, "train.tsv"))
#             random.shuffle(source_train_samples)
#             source_augment_id_ = self.augment_ids[self.source][0]
#             for (i, line) in enumerate(source_train_samples):
#                 sent = line[source_augment_id_]
#                 source_sents.append(sent)
#                 if i > self.n_sample:
#                     break
#         else:
#             # ood sents
#             docs = DocumentDatabase(False)
#             docs.document_shelf = shelve.open('./shelf.db')

#             docs.doc_cumsum = docs.document_shelf["doc_cumsum"]
#             docs.cumsum_max = docs.document_shelf["cumsum_max"]
#             docs.doc_lengths = docs.document_shelf["doc_lengths"] 

#             print(docs.doc_cumsum, docs.document_shelf[str(0)], docs.document_shelf[str(1)])

#             doc_idxs = list( range( len(docs) ) )
#             random.shuffle(doc_idxs)
#             sent_count = 0
#             target_seq_length = max_seq_length - 3

#             doc_database = docs.document_shelf
#             for doc_idx in doc_idxs:
#                 document = doc_database[str(doc_idx)]

#                 i = 0
#                 current_chunk = []
#                 current_length = 0
#                 while i < len(document):
#                     segment = document[i]
#                     current_chunk.append(segment)
#                     current_length += len(segment)
#                     if i == len(document) - 1 or current_length >= target_seq_length:    
#                         if current_chunk:
#                             # `a_end` is how many segments from `current_chunk` go into the `A`
#                             # (first) sentence.
#                             a_end = 1
#                             if len(current_chunk) >= 2:
#                                 a_end = random.randrange(1, len(current_chunk))

#                             tokens_a = []
#                             for j in range(a_end):
#                                 tokens_a.extend(current_chunk[j])

#                             if not tokens_a or len(tokens_a) == 0:
#                                 tokens_a = ["."]
#                             assert len(tokens_a) >= 1

#                             if len(tokens_a) > 5:
#                                 tokens_a = tokens_a[:max_seq_length]
#                                 # print(len(tokens_a))
#                                 source_sents.append( " ".join(tokens_a).replace("##", "") )
#                                 sent_count += 1

#                         current_chunk = []
#                         current_length = 0
#                     i += 1

#                 if sent_count > self.n_sample:
#                     break 
#         return source_sents

#     def read_augment_write(self):
#         task_dir = os.path.join(self.glue_dir, self.task_name)
#         train_samples = _read_tsv(os.path.join(task_dir, "train.tsv"))

#         self.n_sample = len(train_samples) * 1

#         source_sents = self._process_source()
#         logger.info(f"load source_sents!")
#         source = self.source.lower()
#         output_filename = os.path.join(task_dir, f"train_gen_{source}.tsv")

#         augment_ids_ = self.augment_ids[self.task_name]
#         filter_flag = self.filter_flags[self.task_name]

#         pair = len(augment_ids_) == 2

#         label_idx = label_ids[ self.task_name ]
#         with open(output_filename, 'w', newline='', encoding="utf-8") as f:
#             writer = csv.writer(f, delimiter="\t")
#             for (i, line) in enumerate(train_samples):
#                 if i == 0 and filter_flag:
#                     writer.writerow(line)
#                     continue

#                 if pair:
#                     line[augment_ids_[-1]] = source_sents[i]

#                 for augment_id in augment_ids_[:1]:
#                     # sent = line[augment_id]
#                     sent = source_sents[i]

#                     augmented_sents = self.augmentor.augment(sent)
#                     for (augment_sent, augment_label) in augmented_sents:

#                         line[augment_id] = augment_sent
#                         line[ label_idx ] = self.label_list[augment_label]
#                         writer.writerow(line)

#                 if (i+1) % 1000 == 0:
#                     logger.info("Having been processing {} examples".format(str(i+1)))
#                     logger.info(f"label_count {self.augmentor.label_count}")
#         print(self.augmentor.label_count)




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
        batch = [[self.CLS] + [self.MASK] * length + [self.SEP] for _ in range(self.batch_size)]
        return self.tokenize_batch(batch)

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
        # inputs = torch.LongTensor(string_batch).cuda()
        # logits = self.model(inputs)
        logits = self.model(string_batch)
        prob = torch.nn.functional.log_softmax(logits)
        outputs = prob.argmax(dim=1)
        return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None, type=str, 
                        help="Fine-tuned Bert model is under is folder")
    parser.add_argument("--output_dir", default=None, type=str, 
                        help="Generated output file will be placed under this folder")
    parser.add_argument("--generation_mode", default="parallel-sequential", type=str, 
                        help="Mode of generating sentence")
    # parser.add_argument("--task_name", default=None, type=str, 
    #                     help="Which head to use") # sequence classification for now
    parser.add_argument("--max_len", default=50, type=int, 
                        help="Max sequence length to generate")
    parser.add_argument("--iter_num", default=500, type=int, 
                        help="Iteration of repeating masking for each sentence")
    parser.add_argument("--batch_num", default=2, type=int, 
                        help="How many batches to generate")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="How many sentence to generate for one batch")
    parser.add_argument("--top_k", default=100, type=int,
                        help="Choose from top k words instead of full distribution")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--burnin", default=250, type=int,
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
    pretrained_bert_model = f"/rscratch/bohan/ZQBert/zero-shot-qbert/Berts/sst_base_l12/"

    # Prepare two models for generation and labeling
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
    LM_model = BertForMaskedLM.from_pretrained(pretrained_bert_model)
    LM_model.eval()
    LM_model.to(device)

    TA_model = BertForSequenceClassification.from_pretrained(pretrained_bert_model)
    TA_model.eval()
    TA_model.to(device)

    # task_name = args.task_name.lower()
    if args.max_len > args.iter_num:
        logger.info("Iteration less than length of a sentence will possibly result in a sentence with [MASK]")
    
    with torch.no_grad():
        generator = Generate(tokenizer, LM_model, args.generation_mode, args.batch_size, args.max_len, 
                                args.temperature, args.burnin, args.iter_num, args.top_k)
        labeler = Label(tokenizer, TA_model)

        for batch in range(args.batch_num):
            #sentence_batch = generator.generate()
            #print(sentence_batch)
            # print(len(string_batch))

            # TO DO: Take string_batch as input. Do inference and get a label. Write to output tsv.
            #labels = labeler.generate(sentence_batch)
            s = "are more deeply thought through than in most ` right-thinking ' films"
            t = torch.unsqueeze(torch.tensor(tokenizer.encode(s)), 0).to(device)
            new_label = labeler.generate(t)
            print("new_label", new_label)


            # with open('./output_labels.tsv', 'a+') as out_file:
            #     tsv_writer = csv.writer(out_file, delimiter='\t')
            #     for i in range(len(labels)):
            #         sentence = str(string_batch[i])
            #         label = str(int(labels[i].cpu()))
            #         # TODO: replace this line
            #         tsv_writer.writerow([sentence, label])


            logger.info("Having been generating {} batches".format(str(batch+1)))


if __name__ == "__main__":
    main()

