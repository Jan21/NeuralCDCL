import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerFast
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=(data.tokenizer_path)
    )
    tokenizer.eos_token = "[EOS]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):

        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        tokens = torch.tensor(item['tokens'], dtype=torch.long)
        pos_embeddings = torch.tensor(item['pos_embeddings'], dtype=torch.long)
        masks = torch.tensor(item['masks'], dtype=torch.float)
        labels = torch.tensor(item['labels'], dtype=torch.float)
        #keys = torch.tensor(item['keys'], dtype=torch.float)
        #queries = torch.tensor(item['queries'], dtype=torch.float)
        #ids = torch.tensor(item['ids'], dtype=torch.long)

        return {
            'tokens': tokens,
            'pos_embeddings': pos_embeddings,
            'masks': masks,
            'labels': labels,
            #'keys': keys,
            #'queries': queries
        }

class Datamodule(LightningDataModule):
    def __init__(self, datasets, batch_size, val_batch_size, num_workers):
        super(Datamodule, self).__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.datasets['train']
        self.val_dataset = self.datasets['val']
        self.test_dataset = self.datasets['test']

    def collate_fn_pad(self, batch):
        # Extract each component from the batch
        tokens = [item['tokens'] for item in batch]
        pos_embeddings = [item['pos_embeddings'] for item in batch]
        masks = [item['masks'] for item in batch]
        labels = [item['labels'] for item in batch]
        att_mask = [torch.ones(item['tokens'].size(0)) for item in batch]
        #keys = [item['keys'] for item in batch]
        #queries = [item['queries'] for item in batch]
        #ids = [item['ids'] for item in batch]

        # Convert tokens to tensors if they aren't already
        if not isinstance(tokens[0], torch.Tensor):
            tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
        if not isinstance(pos_embeddings[0], torch.Tensor):
            pos_embeddings = [torch.tensor(p, dtype=torch.long) for p in pos_embeddings]
        if not isinstance(masks[0], torch.Tensor):
            masks = [torch.tensor(m, dtype=torch.float) for m in masks]
        if not isinstance(labels[0], torch.Tensor):
            labels = [torch.tensor(l, dtype=torch.float) for l in labels]
        #if not isinstance(ids[0], torch.Tensor):
        #    ids = [torch.tensor(i, dtype=torch.long) for i in ids]

        # Get max lengths for padding
        max_token_len = max(t.size(0) for t in tokens)
        max_pos_toks = max(p.size(0) for p in pos_embeddings)
        # Pad sequences
        tpad = torch.nn.functional.pad
        padded_tokens = torch.stack([tpad(t, (0, max_token_len - t.size(0)), value=290) for t in tokens]) # TODO fix this 225
        padded_pos = torch.stack([tpad(p, (0,max_token_len  - p.size(1),0,max_pos_toks - p.size(0)), value=290) for p in pos_embeddings])
        padded_masks = torch.stack([tpad(m, (0, max_token_len - m.size(1),0,max_pos_toks - m.size(0)), value=0) for m in masks])
        padded_labels = torch.stack(labels)
        padded_att_mask = torch.stack([tpad(a, (0, max_token_len - a.size(0)), value=0) for a in att_mask])
        #padded_keys = torch.stack([tpad(k, (0,0,0, max_token_len - k.size(0)), value=0) for k in keys])
        #padded_queries = torch.stack([tpad(q, (0,0,0, max_token_len - q.size(0)), value=0) for q in queries])

        return {
            'tokens': padded_tokens,
            'pos_embeddings': padded_pos,
            'masks': padded_masks,
            'labels': padded_labels,
            'att_masks': padded_att_mask,
            #'keys': padded_keys,
            #'queries': padded_queries
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad
        )

def get_data(data: DictConfig):

    with open(data.datapath_train, "rb") as f:
        train = pickle.load(f)
    with open(data.datapath_val, "rb") as f:
        test = pickle.load(f)

    tokenizer = get_tokenizer(data)
    
    processed_train = {}
    for idx in train:
        processed_train[idx] = train[idx].copy()
        #toks = []
        
        #for i in train[idx]["tokens"]:
        #    toks.append(tokenizer.encode(i)[0])
        #processed_train[idx]['tokens'] = toks
        #pos_mat = []
        #for row in train[idx]["pos_embeddings"]:
        #    pos_row = []
        #    for i in row:
        #        pos_row.append(tokenizer.encode(i)[0])
        #    pos_mat.append(pos_row)
        #processed_train[idx]['pos_embeddings'] = pos_mat
        #ids = []
        #for i in train[idx]["ids"]:
        #    ids.append(tokenizer.encode(i)[0])
        #processed_train[idx]['ids'] = ids
    
    processed_test = {}
    for idx in test:
        if idx > 50:
            break
        processed_test[idx] = test[idx].copy() 

        #toks = []
        #for i in test[idx]["tokens"]:
        #    toks.append(tokenizer.encode(i)[0])
        
        #processed_test[idx]['tokens'] = toks
        #pos_mat = []
        #for row in test[idx]["pos_embeddings"]:
        #    pos_row = []
        #    for i in row:
        #        pos_row.append(tokenizer.encode(i)[0])
        #    pos_mat.append(pos_row)
        #processed_test[idx]['pos_embeddings'] = pos_mat
        #ids = []
        #for i in test[idx]["ids"]:
        #    ids.append(tokenizer.encode(i)[0])
        #processed_test[idx]['ids'] = ids

    train_dataset = SequenceDataset(processed_train)
    test_dataset = SequenceDataset(processed_test)
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'val': test_dataset
    }