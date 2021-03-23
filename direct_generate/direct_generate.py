"""
Generate sentence and label directly from fine-tuned Bert.
Idea from "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model" (https://arxiv.org/pdf/1902.04094.pdf).
"""

import random
import sys
import os
import unicodedata
import re
import logging
import csv
import argparse


import torch
import numpy as np

import json
import collections
import logging
import os
import shelve
from multiprocessing import Pool

from ood_tmp_doc_db import DocumentDatabase


import sys
sys.path.append("..")
from transformer import BertTokenizer, BertForMaskedLM
from transformer.modeling import TinyBertForSequenceClassification
from run_glue import (MrpcProcessor, ColaProcessor, MnliProcessor, MnliMismatchedProcessor, 
    Sst2Processor, StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor)

processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_length = 128
StopWordsList = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                 "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "'s", "'re"]


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):
        # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


# valid string only includes al
def _is_valid(string):
    return True if not re.search('[^a-z]', string) else False


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def prepare_embedding_retrieval(glove_file, vocab_size=100000):
    cnt = 0
    words = []
    embeddings = {}

    # only read first 100,000 words for fast retrieval
    with open(glove_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split()
            words.append(items[0])
            embeddings[items[0]] = [float(x) for x in items[1:]]

            cnt += 1
            if cnt == vocab_size:
                break

    vocab = {w: idx for idx, w in enumerate(words)}
    ids_to_tokens = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(embeddings[ids_to_tokens[0]])
    emb_matrix = np.zeros((vocab_size, vector_dim))
    for word, v in embeddings.items():
        if word == '<unk>':
            continue
        emb_matrix[vocab[word], :] = v

    # normalize each word vector
    d = (np.sum(emb_matrix ** 2, 1) ** 0.5)
    emb_norm = (emb_matrix.T / d).T
    return emb_norm, vocab, ids_to_tokens

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class DataAugmentor(object):
    def __init__(self, model, tokenizer, teacher_model, teacher_model_tokenizer, emb_norm, vocab, ids_to_tokens, 
        M, N, p, p_dist=0.5, pair=False):
        self.model = model
        self.tokenizer = tokenizer
        self.emb_norm = emb_norm
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.M = M
        self.N = N
        self.p = p
        self.p_dist = p_dist
        self.teacher_model = teacher_model
        self.teacher_model_tokenizer = teacher_model_tokenizer
        self.num_labels = teacher_model.num_labels
        self.label_count = [0 for _ in range(self.num_labels)]
        self.pair = pair

    def _word_distance(self, word):
        p_dist = random.random() < self.p_dist
        if word not in self.vocab.keys():
            return []
        word_idx = self.vocab[word]
        word_emb = self.emb_norm[word_idx]

        dist = np.dot(self.emb_norm, word_emb.T)
        dist[word_idx] = -np.Inf

        if p_dist:
            candidate_ids = np.argsort(-dist)[:self.M]
        else:
            candidate_ids = np.argsort(dist)[:self.M]

        return [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

    def _check_label(self, sent, candidate_sent):
        tokenized_text = self.teacher_model_tokenizer.tokenize(candidate_sent)
        

        if self.pair:
            tokenized_text_a = self.teacher_model_tokenizer.tokenize(sent)
            _truncate_seq_pair(tokenized_text_a, tokenized_text, max_seq_length - 3, )
            # tokenized_text = tokenized_text_a + ['[SEP]'] + tokenized_text + ['[SEP]']
        else:
            tokenized_text_a = None
            if len(tokenized_text) > max_seq_length - 2:
                tokenized_text = tokenized_text[:(max_seq_length - 2)]
            # tokenized_text = tokenized_text + ['[SEP]']

        tokens = ["[CLS]"] + tokenized_text + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokenized_text_a:
            tokens += tokenized_text_a + ["[SEP]"]
            segment_ids += [1] * (len(tokenized_text_a) + 1)

        input_ids = self.teacher_model_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        
        segments_tensor = torch.tensor([segment_ids]).to(device)
        tokens_tensor = torch.tensor([input_ids]).to(device)
        input_mask_tensor = torch.tensor([input_mask]).to(device)

        self.teacher_model.to(device)
        logits, _, _ = self.teacher_model(tokens_tensor, segments_tensor, input_mask_tensor, is_student=True)
        label = np.argmax( logits.cpu().numpy()[0] )
        return label

    def _masked_language_model(self, sent, word_pieces, mask_id):
        tokenized_text = self.tokenizer.tokenize(sent)
        tokenized_text = ['[CLS]'] + tokenized_text
        tokenized_len = len(tokenized_text)

        tokenized_text = word_pieces + ['[SEP]'] + tokenized_text[1:] + ['[SEP]']

        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:512]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) - tokenized_len - 1)

        tokens_tensor = torch.tensor([token_ids]).to(device)
        segments_tensor = torch.tensor([segments_ids]).to(device)

        self.model.to(device)

        predictions = self.model(tokens_tensor, segments_tensor)

        word_candidates = torch.argsort(predictions[0, mask_id], descending=True)[:self.M].tolist()
        word_candidates = self.tokenizer.convert_ids_to_tokens(word_candidates)

        return list(filter(lambda x: x.find("##"), word_candidates))

    def _word_augment(self, sentence, mask_token_idx, mask_token):
        word_pieces = self.tokenizer.tokenize(sentence)
        word_pieces = ['[CLS]'] + word_pieces
        tokenized_len = len(word_pieces)

        token_idx = -1
        for i in range(1, tokenized_len):
            if "##" not in word_pieces[i]:
                token_idx = token_idx + 1
                if token_idx < mask_token_idx:
                    word_piece_ids = []
                elif token_idx == mask_token_idx:
                    word_piece_ids = [i]
                else:
                    break
            else:
                word_piece_ids.append(i)
        if mask_token == '[UNK]':
            print(word_piece_ids, len(word_piece_ids))
            exit()

        if len(word_piece_ids) == 1:
            word_pieces[word_piece_ids[0]] = '[MASK]'
            candidate_words = self._masked_language_model(
                sentence, word_pieces, word_piece_ids[0])
        elif len(word_piece_ids) > 1:
            candidate_words = self._word_distance(mask_token)
        else:
            logger.info("invalid input sentence!")
        
        if len(candidate_words)==0:
            candidate_words.append(mask_token)

        return candidate_words

    def augment(self, sent):
        # candidate_sents = [sent]

        candidate_sents = []

        tokens = self.tokenizer.basic_tokenizer.tokenize(sent)
        candidate_words = {}
        for (idx, word) in enumerate(tokens):
            if _is_valid(word) and word not in StopWordsList:
                candidate_words[idx] = self._word_augment(sent, idx, word)
        # logger.info(candidate_words)
        cnt = 0
        while cnt < self.N:
            new_sent = list(tokens)
            for idx in candidate_words.keys():
                candidate_word = random.choice(candidate_words[idx])

                x = random.random()
                if x < self.p:
                    new_sent[idx] = candidate_word

            if " ".join(new_sent) not in candidate_sents:
                label = self._check_label( sent, " ".join(new_sent),  )

                candidate_sents.append( (' '.join(new_sent), label) )

                self.label_count[label] += 1

            cnt += 1

        return candidate_sents

augment_ids = {'MRPC': [3, 4], 'MNLI': [8, 9], 'CoLA': [3], 'SST-2': [0],
                'STS-B': [7, 8], 'QQP': [3, 4], 'QNLI': [1, 2], 'RTE': [1, 2]}

label_ids = {'MRPC': 0, 'MNLI': -1, 'CoLA': 1, 'SST-2': 1,
            'STS-B': -1, 'QQP': 5, 'QNLI': -1, 'RTE': -1}

class AugmentProcessor(object):
    def __init__(self, augmentor, glue_dir, task_name, source="MNLI", label_list=None):
        self.augmentor = augmentor
        self.glue_dir = glue_dir
        self.task_name = task_name

        self.source = source
        self.augment_ids = {'MRPC': [3, 4], 'MNLI': [8, 9], 'CoLA': [3], 'SST-2': [0],
                            'STS-B': [7, 8], 'QQP': [3, 4], 'QNLI': [1, 2], 'RTE': [1, 2]}

        self.filter_flags = { 'MRPC': True, 'MNLI': True, 'CoLA': False, 'SST-2': True,
                              'STS-B': True, 'QQP': True, 'QNLI': True, 'RTE': True}
        self.n_sample = 10000
        self.label_list = label_list

        assert self.task_name in self.augment_ids

    def _process_source(self):
        source_sents = []
        if self.source in self.augment_ids:
            source_dir = os.path.join(self.glue_dir, self.source)
            source_train_samples = _read_tsv(os.path.join(source_dir, "train.tsv"))
            random.shuffle(source_train_samples)
            source_augment_id_ = self.augment_ids[self.source][0]
            for (i, line) in enumerate(source_train_samples):
                sent = line[source_augment_id_]
                source_sents.append(sent)
                if i > self.n_sample:
                    break
        else:
            # ood sents
            docs = DocumentDatabase(False)
            docs.document_shelf = shelve.open('./shelf.db')

            docs.doc_cumsum = docs.document_shelf["doc_cumsum"]
            docs.cumsum_max = docs.document_shelf["cumsum_max"]
            docs.doc_lengths = docs.document_shelf["doc_lengths"] 

            print(docs.doc_cumsum, docs.document_shelf[str(0)], docs.document_shelf[str(1)])

            doc_idxs = list( range( len(docs) ) )
            random.shuffle(doc_idxs)
            sent_count = 0
            target_seq_length = max_seq_length - 3

            doc_database = docs.document_shelf
            for doc_idx in doc_idxs:
                document = doc_database[str(doc_idx)]

                i = 0
                current_chunk = []
                current_length = 0
                while i < len(document):
                    segment = document[i]
                    current_chunk.append(segment)
                    current_length += len(segment)
                    if i == len(document) - 1 or current_length >= target_seq_length:    
                        if current_chunk:
                            # `a_end` is how many segments from `current_chunk` go into the `A`
                            # (first) sentence.
                            a_end = 1
                            if len(current_chunk) >= 2:
                                a_end = random.randrange(1, len(current_chunk))

                            tokens_a = []
                            for j in range(a_end):
                                tokens_a.extend(current_chunk[j])

                            if not tokens_a or len(tokens_a) == 0:
                                tokens_a = ["."]
                            assert len(tokens_a) >= 1

                            if len(tokens_a) > 5:
                                tokens_a = tokens_a[:max_seq_length]
                                # print(len(tokens_a))
                                source_sents.append( " ".join(tokens_a).replace("##", "") )
                                sent_count += 1

                        current_chunk = []
                        current_length = 0
                    i += 1

                if sent_count > self.n_sample:
                    break 
        return source_sents

    def read_augment_write(self):
        task_dir = os.path.join(self.glue_dir, self.task_name)
        train_samples = _read_tsv(os.path.join(task_dir, "train.tsv"))

        self.n_sample = len(train_samples) * 1

        source_sents = self._process_source()
        logger.info(f"load source_sents!")
        source = self.source.lower()
        output_filename = os.path.join(task_dir, f"train_gen_{source}.tsv")

        augment_ids_ = self.augment_ids[self.task_name]
        filter_flag = self.filter_flags[self.task_name]

        pair = len(augment_ids_) == 2

        label_idx = label_ids[ self.task_name ]
        with open(output_filename, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for (i, line) in enumerate(train_samples):
                if i == 0 and filter_flag:
                    writer.writerow(line)
                    continue

                if pair:
                    line[augment_ids_[-1]] = source_sents[i]

                for augment_id in augment_ids_[:1]:
                    # sent = line[augment_id]
                    sent = source_sents[i]

                    augmented_sents = self.augmentor.augment(sent)
                    for (augment_sent, augment_label) in augmented_sents:

                        line[augment_id] = augment_sent
                        line[ label_idx ] = self.label_list[augment_label]
                        writer.writerow(line)

                if (i+1) % 1000 == 0:
                    logger.info("Having been processing {} examples".format(str(i+1)))
                    logger.info(f"label_count {self.augmentor.label_count}")
        print(self.augmentor.label_count)

class Generate(object):
    def __init__(self, tokenizer, LM_model):
        self.tokenizer = tokenizer
        self.model = LM_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None, type=str, 
                        help="Fine-tuned Bert model is under is folder")
    parser.add_argument("--output_dir", default=None, type=str, 
                        help="Generated output will be placed under this folder")
    parser.add_argument("--generation_mode", default="parallel-sequential", type=str, 
                        help="Mode of generating sentence")
    # parser.add_argument("--task_name", default=None, type=str, 
    #                     help="Which processor to use")
    parser.add_argument("--max_len", default=512, type=int, 
                        help="Max sequence length to generate")
    parser.add_argument("--bytes", default="1M", type=str, 
                        help="Size of data to generate (default %(default)s)", 
                        metavar="n[KMG]")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed to reproduce results")

    # original argument
    # parser.add_argument("--pretrained_bert_model", default="bert_base", type=str, 
    #                     help="Downloaded pretrained model (bert-base-uncased) is under this folder")
    # parser.add_argument("--glove_embs", default="/rscratch/sheng.s/clip_boi/grid-feats-vqa/mmf/glove.6B.300d.txt", type=str, 
    #                     help="Glove word embeddings file")
    # parser.add_argument("--glue_dir", default="/dnn/sheng.s/glue_data/", type=str,
    #                     help="GLUE data dir")
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="Task(eg. CoLA, SST-2) that we want to do data augmentation for its train set")
    # parser.add_argument("--N", default=2, type=int,
    #                     help="How many times is the corpus expanded?")
    # parser.add_argument("--M", default=15, type=int,
    #                     help="Choose from M most-likely words in the corresponding position")
    # parser.add_argument("--p", default=0.15, type=float,
    #                     help="Threshold probability p to replace current word")

    # parser.add_argument("--p_dist", default=0.5, type=float,
    #                     help="Threshold probability p_dist to replace current word by the most / least likely word")

    # parser.add_argument("--source", default="MNLI", type=str,
    #                     help="source to generate the current pesudo label")
    
    # parser.add_argument("--label_balance", default=True, type=bool,
    #                     help="do we ensure the label balance in generate dataset")


    args = parser.parse_args()
    # logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # TO DO: set fine-tuned model for now, later take from arg
    pretrained_bert_model = f"/rscratch/bohan/ZQBert/zero-shot-qbert/Berts/mrpc_base_l12/"
    # Prepare two models for generation and labeling
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
    LM_model = BertForMaskedLM.from_pretrained(pretrained_bert_model)
    # TO DO: How to load fine-tuned model 
    TA_model = 
    LM_model.eval()
    # TA_model = 

    # task_name = args.task_name.lower()
    # processor = processors[task_name]()
    # label_list = processor.get_labels()


    # pair = len(augment_ids[args.task_name]) == 2
    
    with torch.no_grad():
        # data_augmentor = DataAugmentor(model, tokenizer, teacher_model, teacher_model_tokenizer, 
        #     emb_norm, vocab, ids_to_tokens, args.M, args.N, args.p, p_dist=args.p_dist, pair=pair)

        # # Do data augmentation
        # if args.task_name == "MNLI":
        #     assert args.source != "MNLI"

        # processor = AugmentProcessor(data_augmentor, args.glue_dir, args.task_name, args.source, label_list=label_list)
        # processor.read_augment_write()


if __name__ == "__main__":
    main()

