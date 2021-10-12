import torch
import copy
from torch import nn
import numpy as np
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertLayer
from transformers import (BertConfig, BertTokenizer, BertForQuestionAnswering,
                          RobertaConfig, RobertaTokenizer, RobertaForQuestionAnswering,
                          AlbertConfig, AlbertTokenizer, AlbertForQuestionAnswering)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}

IGNORE_INDEX = -100
dropout = torch.nn.Dropout(0.1)
                                                                                                                                                                                                        
Q_type = {"compositional":0, "inference":1, "comparison":2, "bridge_comparison":3}
relations = ['date of birth', 'publisher', 'child', 'director', 'composer',
            'doctoral advisor', 'mother', 'place of death', 'award received',
            'sibling', 'father', 'occupation', 'educated at', 'country of citizenship',
            'spouse', 'creator', 'cause of death', 'country of origin', 'performer',
            'employer', 'place of birth', 'place of detention', 'student of', 'publication date',
            'place of burial', 'inception', 'has part', 'manufacturer', 'presenter', 'country',
            'founded by', 'editor', 'producer', 'date of death', 'Birth-Death']
r_dict = {'date of birth': 0, 'publisher': 1, 'child': 2, 'director': 3, 'composer': 4,
            'doctoral advisor': 5, 'mother': 6, 'place of death': 7, 'award received': 8,
            'sibling': 9, 'father': 10, 'occupation': 11, 'educated at': 12,
            'country of citizenship': 13, 'spouse': 14, 'creator': 15, 'cause of death': 16,
            'country of origin': 17, 'performer': 18, 'employer': 19, 'place of birth': 20,
            'place of detention': 21, 'student of': 22, 'publication date': 23,
            'place of burial': 24, 'inception': 25, 'has part': 26, 'manufacturer': 27,
            'presenter': 28, 'country': 29, 'founded by': 30, 'editor': 31, 'producer': 32,
            'date of death': 33, 'Birth-Death': 34}                                                                                                                                                        

                                                                                                                                                                                                           
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
                                                                                                                                                                                                           
                                                                                                                                                                                                           
def add_figure(name, writer, global_step, train_loss):
    writer.write(name + '_data/train_loss: {} \t global_step:{} \n'.format(train_loss, global_step))
    return
    
def find_pos(context, ans):
    context = context.replace(ans,'{'+ans+'}').split()
    y1,y2 = 0,0
    for i in range(len(context)):
        if '{' in context[i]:
            y1 = i
        if '}' in context[i]:
            y2 = i
            break
    return y1,y2

from collections import Counter
def sim_score(a,b):
    common = Counter(a) & Counter(b)
    correct = sum(common.values())
    if correct==0:
        return 0.0
    pred,recall = correct/len(a),correct/len(b)
    return 2*pred*recall/(pred+recall)
    
def find_subpos(context, ans):
    context = context.replace('##','').split()
    ans = ans.replace('##','').split()
    ra,rb,ms = 0,0,0.0
    for i in range(len(context)):
        for j in range(i+1, min(len(context),i+len(ans)+1)):
            score = sim_score(ans, context[i:j])
            if score>ms:
                ra,rb,ms = i,j-1,score
    return ra,rb

def F1(a,b,c,d):
    correct = max([0,min(b,d)-max(a,c)])
    if correct == 0:
        return 0.0,0.0,0.0
    pred, recall = correct/(b-a), correct/(d-c)
    return 2*pred*recall/(pred+recall), pred, recall

def FindSpan(start,end):
    d = len(start)
    adj = start.reshape(-1,1)*end.reshape(1,-1)
    adj = np.triu(adj).reshape(-1)
    index = int(np.argmax(adj))
    return int(index/d), index%d


class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, mask):
        """
        :param doc_state:  N x L x d
        :param entity_lens:  N x L
        :return: N x 2d
        """
        max_pooled = torch.max(doc_state, dim=1)[0]
        mean_pooled = torch.sum(doc_state, dim=1) / torch.sum(mask, dim=1).unsqueeze(1)
        output = torch.cat([max_pooled, mean_pooled], dim=1)  # N x 2d
        return output

class BertForCompMhop(BertPreTrainedModel):
                                                                                                                                                                                                           
    def __init__(self, config, num_labels=35):
        super(BertForCompMhop, self).__init__(config)
        self.num_labels = num_labels                                                                                                                                                                       
        self.bert = BertModel(config)
        self.encoder = BertLayer(config)
        self.pooler = MeanMaxPooling()                                                                                                                                                                   
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.TypeLinear = torch.nn.Linear(config.hidden_size*2, 4)
        self.R1Linear = torch.nn.Linear(config.hidden_size*2, 4*num_labels)
        self.R2Linear = torch.nn.Linear(config.hidden_size*2, 4*num_labels)
        self.init_weights()
                                                                                                                                                                                                           
    def forward(self, batch, is_train = False):
        input_ids, attention_mask = batch['context_idxs'], batch['context_mask']
        embed_output = self.bert(input_ids, None, attention_mask)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_output = self.encoder(embed_output[0], extended_attention_mask)
        pooled_output = self.dropout(self.pooler(encoder_output[0], attention_mask))                                                                                                                                          
        t_logits = self.TypeLinear(pooled_output)
        r1_logits = self.R1Linear(pooled_output)
        r2_logits = self.R2Linear(pooled_output)
                                                                                                                                                                                                           
        r1 = torch.matmul(r1_logits.view(-1, self.num_labels, 4), t_logits.view(-1,4,1)).view(-1,self.num_labels)
        r2 = torch.matmul(r2_logits.view(-1, self.num_labels, 4), t_logits.view(-1,4,1)).view(-1,self.num_labels)
        if is_train:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(t_logits, batch['y']) + loss_fct(r1, batch['y1']) + loss_fct(r2, batch['y2'])
            return loss
        else:
            return t_logits, r1, r2

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = [batch, len, h1]
        y = [batch, h2]
        x_mask = [batch, len]
        """
        x = dropout(x)
        y = dropout(y)

        Wy = self.linear(y) if self.linear is not None else y
        #xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy = torch.matmul(x, Wy.unsqueeze(2)).squeeze(2)
        xWy = xWy + (1.0-x_mask)*(-10000.0)
        return xWy # [batch,len]
                
class GetSpanStartEnd(nn.Module):
    # supports MLP attention and GRU for pointer network updating
    def __init__(self, x_size, h_size, do_indep_attn=True, attn_type="Bilinear", do_ptr_update=True):
        super(GetSpanStartEnd, self).__init__()

        self.attn  = BilinearSeqAttn(x_size, h_size)
        self.attn2 = BilinearSeqAttn(x_size, h_size) if do_indep_attn else None

        self.rnn = nn.GRUCell(x_size, h_size) if do_ptr_update else None

    # x -- doc_hiddens [10,384,250]
    # h0 -- question_avg_hidden [10,125]
    # x_mask [10,384]
    def forward(self, x, h0, x_mask):
        """
        x = [batch, len, x_hidden_size]
        h0 = [batch, h_size]
        x_mask = [batch, len]
        """
        start_scores = self.attn(x, h0, x_mask) # [10,384]
        # start_scores [batch, len]

        if self.rnn is not None:
            ptr_net_in = torch.bmm(torch.nn.functional.softmax(start_scores, dim=1).unsqueeze(1), x).squeeze(1) # [10,250]
            ptr_net_in = dropout(ptr_net_in)
            h0 = dropout(h0)
            h1 = self.rnn(ptr_net_in, h0) # [10,125]
            # h1 same size as h0
        else:
            h1 = h0

        end_scores = self.attn(x, h1, x_mask) if self.attn2 is None else\
                     self.attn2(x, h1, x_mask)
        # end_scores = batch * len
        return start_scores, end_scores # [10,384]

                                                                                                                                                                                                         
class BertForTypeSpanComp(BertPreTrainedModel):
                                                                                                                                                                                                           
    def __init__(self, config):
        super(BertForTypeSpanComp, self).__init__(config)                                                                                                                                                                      
        self.bert = BertModel(config)
        self.encoder = BertLayer(config)
        self.pooler = MeanMaxPooling()                                                                                                                                                                   
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.TypeLinear = torch.nn.Linear(config.hidden_size*2, 4)
        ptrnet = GetSpanStartEnd(config.hidden_size,config.hidden_size*2)
        self.RelaPointer = nn.ModuleList([copy.deepcopy(ptrnet) for _ in range(8)])
        self.init_weights()
                                                                                                                                                                                                           
    def forward(self, batch, is_train = False):
        input_ids, attention_mask = batch['context_idxs'], batch['context_mask']
        bsz = input_ids.size(0)
        embed_output = self.bert(input_ids, None, attention_mask)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_output = self.encoder(embed_output[0], extended_attention_mask)[0]
        pooled_output = self.dropout(self.pooler(encoder_output,attention_mask))                                                                                                                                          
        t_logits = self.TypeLinear(pooled_output)
        R_logits = []
        for layer_module in self.RelaPointer:
            start,end = layer_module(encoder_output, pooled_output, attention_mask)
            R_logits += [start,end]
        R_logits = torch.cat(R_logits,-1)
        r_logits = torch.matmul(R_logits.view(bsz, -1, 4), t_logits.view(-1,4,1)).view(bsz, -1, 4)
                                                                                                                                                                                                           
        r1s,r1e,r2s,r2e = r_logits.split(1, dim=-1)
        r1s,r1e,r2s,r2e = r1s.squeeze(-1), r1e.squeeze(-1), r2s.squeeze(-1), r2e.squeeze(-1)
        
        if is_train:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss1 = loss_fct(r1s, batch['y1s']) + loss_fct(r1e, batch['y1e'])
            loss2 = loss_fct(r2s, batch['y2s']) + loss_fct(r2e, batch['y2e'])
            loss = loss_fct(t_logits, batch['y']) + loss1 + loss2
            return loss, loss1, loss2
        else:
            return t_logits, r1s,r1e,r2s,r2e

                                                                                                                                                                                                           
class BertForSpanComp(BertPreTrainedModel):
                                                                                                                                                                                                           
    def __init__(self, config):
        super(BertForSpanComp, self).__init__(config)                                                                                                                                                                     
        self.bert = BertModel(config)                                                                                                                                                                      
        self.pooler = MeanMaxPooling()                                                                                                                                                                   
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.TypeLinear = torch.nn.Linear(config.hidden_size, 4)
        self.RelaPointer1 = GetSpanStartEnd(config.hidden_size,config.hidden_size)
        self.RelaPointer2 = GetSpanStartEnd(config.hidden_size,config.hidden_size)
        self.init_weights()
                                                                                                                                                                                                           
    def forward(self, batch, is_train = False):
        input_ids, attention_mask = batch['context_idxs'], batch['context_mask']
        encoder_output = self.bert(input_ids, None, attention_mask)
        pooled_output = self.dropout(encoder_output[1])
        encoder_output = encoder_output[0]
            
        t_logits = self.TypeLinear(pooled_output)
        r1s,r1e = self.RelaPointer1(encoder_output, pooled_output, attention_mask)
        r2s,r2e = self.RelaPointer2(encoder_output, pooled_output, attention_mask)
        
        if is_train:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss1 = loss_fct(r1s, batch['y1s']) + loss_fct(r1e, batch['y1e'])
            loss2 = loss_fct(r2s, batch['y2s']) + loss_fct(r2e, batch['y2e'])
            loss = loss_fct(t_logits, batch['y']) + loss1 + loss2
            return loss, loss1, loss2
        else:
            return t_logits, r1s,r1e,r2s,r2e
            

class BertForCompare(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super(BertForCompare, self).__init__(config)
        self.num_labels = num_labels                                                                                                                                                                       
        self.bert = BertModel(config)
        self.encoder = BertLayer(config)
        
        self.pooler = MeanMaxPooling()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.CompLinear = torch.nn.Linear(config.hidden_size*2, 4)
        self.init_weights()                                                                                                                                                                                               
    def forward(self, batch, is_train = False):
        input_ids, attention_mask, segment_idxs = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        embed_output = self.bert(input_ids, segment_idxs, attention_mask)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_output = self.encoder(embed_output[0], extended_attention_mask)[0]
        pooled_output = self.dropout(self.pooler(encoder_output,attention_mask))                                                                                                                                          
        logits = self.CompLinear(pooled_output)
        
        if is_train:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, batch['y'])
            return loss
        else:
            return logits