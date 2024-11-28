"""
NOTE: authors forgot to add this. We retrieved it from another repo of theirs at:
https://github.com/fengranMark/ConvRelExpand/blob/main/scripts/data_structure.py
We are still not sure if this is the supposed class, but at least the input/output arguments now match.
"""
import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
from torch.utils.data import IterableDataset
import torch.distributed as dist

from utils import parse_relevant_ids, is_relevant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number



@dataclass
class RewriteSample:
    sample_id: str
    rewrite: str


class ANCERewriteDataset(Dataset):
    def __init__(self, args, query_tokenizer, filename):
        self.examples = []

        relevant_ids = parse_relevant_ids(args)

        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)

        logging.info("Loading {} data file...".format(filename))

        for i in trange(n):
            data[i] = json.loads(data[i])
            if 'id' in data[i]:
                sample_id = data[i]['id']
            else:
                sample_id = data[i]['sample_id']

            if not is_relevant(sample_id, relevant_ids):
                continue

            if 'output' in data[i]:
                rewrite = data[i]['output']
            elif 'rewrite_utt_text' in data[i]:
                rewrite = data[i]['rewrite_utt_text']
            else:
                rewrite = data[i]['oracle_utt_text']

            rewrite = query_tokenizer.encode(rewrite, add_special_tokens=True)

            self.examples.append(RewriteSample(sample_id, rewrite)) 


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_rewrite":[],
                "bt_rewrite_mask":[],
            }
            
            bt_sample_id = [] 
            bt_rewrite = []
            bt_rewrite_mask = []

            for example in batch:
                # padding
                rewrite, rewrite_mask = pad_seq_ids_with_mask(example.rewrite, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                bt_rewrite.append(rewrite)
                bt_rewrite_mask.append(rewrite_mask)     

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_rewrite"] = bt_rewrite
            collated_dict["bt_rewrite_mask"] = bt_rewrite_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
    

def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[-max_length:]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask
