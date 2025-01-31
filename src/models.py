import sys

sys.path += ['../']
import torch
from torch import nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config, T5EncoderModel,
                          GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel,
                          BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, BertForTokenClassification,
                          DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
                          DPRContextEncoder, DPRQuestionEncoder)

import torch.nn.functional as F
from IPython import embed
import time


# ANCE model
class ANCE(RobertaForSequenceClassification):
    # class Pooler:   # adapt to DPR
    #     def __init__(self, pooler_output):
    #         self.pooler_output = pooler_output

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)

class BERT(BertForSequenceClassification):
    # class Pooler:   # adapt to DPR
    #     def __init__(self, pooler_output):
    #         self.pooler_output = pooler_output

    def __init__(self, config):
        BertForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)

class ANCE_Filter(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.apply(self._init_weights)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.use_mean = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def forward(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        logits = self.classifier(full_emb)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities

class Bert_Filter(BertForSequenceClassification):
    def __init__(self, config):
        BertForSequenceClassification.__init__(self, config)
        self.apply(self._init_weights)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.use_mean = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def forward(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        logits = self.classifier(full_emb)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities

class Bert_Quretec(BertForTokenClassification):
    def __init__(self, config):
        BertForTokenClassification.__init__(self, config)
        self.apply(self._init_weights)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels) # 2


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs1.last_hidden_state
        #loss = outputs1.loss
        loss = None
        logits = self.classifier(last_hidden_state)
        probabilities = nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_func(active_logits, active_labels)
            else:
                loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
 
        return loss, logits, probabilities

class ANCE_Multi(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.use_mean = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def forward(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query = self.norm(self.embeddingHead(full_emb))
        logits = self.classifier(full_emb)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities, query


class T5(nn.Module):
    def __init__(self, config, model_path):
        super(T5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_path)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask, labels):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        decode_loss = outputs.loss
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        full_emb = self.masked_mean_or_first(encoder_last_hidden_state, attention_mask)
        query = self.norm(self.embeddingHead(full_emb))
        return query, decode_loss


    def passage_emb(self, input_ids, attention_mask, labels=None):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        full_emb = self.masked_mean_or_first(encoder_last_hidden_state, attention_mask)
        passage = self.norm(self.embeddingHead(full_emb))
        return passage
    

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, labels, mode):
        if mode == "query":
            return self.query_emb(input_ids, attention_mask, labels)
        elif mode == "passage":
            return self.passage_emb(input_ids, attention_mask, labels)

'''
Model-related functions
'''

def load_model(model_type, model_path):
    if model_type == "ANCE_Query" or model_type == "ANCE_Passage":
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE.from_pretrained(model_path, config=config)
    elif model_type == "ANCE_Multi":
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE_Multi.from_pretrained(model_path, config=config)
    elif model_type == "T5_Query" or model_type == "T5_Passage":
        config = T5Config.from_pretrained(
            model_path,
        )
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = T5(config=config, model_path=model_path)
    elif model_type == "BERT_Query" or model_type == "BERT_Passage":
        config = BertConfig.from_pretrained(
            model_path,
        )
        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = BERT.from_pretrained(model_path, config=config)
    elif model_type == "ANCE_Filter":
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE_Filter.from_pretrained(model_path, config=config)
    elif model_type == "BERT_Filter":
        config = BertConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = Bert_Filter.from_pretrained(model_path, config=config)
    elif model_type == "BERT_Quretec":
        config = BertConfig.from_pretrained(
            model_path,
        )
        config.num_labels = 2
        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = Bert_Quretec.from_pretrained(model_path, config=config)

    else:
        raise ValueError
    return tokenizer, model
