from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import logging
import argparse
from tqdm import tqdm, trange
from collections import Counter
                                                                                                                                                                                                           
import random
import ujson as json
                                                                                                                                                                                                           
from func import *


def distance(text, ori_text):
    score = 0.0
    i,j = 0,0
    while i < len(text) and j < len(ori_text):
        if text[i] == ori_text[j]:
            score += 1.0
            i,j = i+1,j+1
        elif text[i] in ori_text[j]:
            score += 0.5
            i,j = i+1,j+1
        else:
            j += 1
    if i < len(text):
        return -1
    return score/len(ori_text)
    

def get_best_answer(text, ori_text):
    pre_ori = [pro_sent(x) for x in ori_text]
    if len(text)==1:
        candidate = []
        for i in range(len(ori_text)):
            if text[0] == pre_ori[i]:
                return ori_text[i]
            if text[0] in pre_ori[i]:
                candidate.append(ori_text[i])
        if len(candidate)==0:
            return ""
        candidate.sort(key=lambda s:len(s))
        return candidate[0]
    start,end = [],[]
    for i in range(len(ori_text)):
        if text[0] in pre_ori[i]:
            start.append(i)
        if text[-1] in pre_ori[i]:
            end.append(i)
    candidate = []
    for i in start:
        for j in end:
            if i>=j or len(text) > j-i+1:
                continue
            score = distance(text, pre_ori[i:j+1])
            if score>0:
                candidate.append((i,j,score))
    if len(candidate) == 0:
        return ""
    candidate.sort(key=lambda x:-x[2])
    return " ".join(ori_text[candidate[0][0]:candidate[0][1]+1])
    

def Predict(reader, compartor, examples, questions, Qtypes, prediction_file, config):
    answer_dict, sp_dict, evidence_dict, all_answer = {},{},{},{}
    for item in tqdm(questions):
        ids, question = item['_id'],item['question']
        titles = examples[ids]['title']
        sort_map = examples[ids]['sort_map']
        Type = Qtypes[ids]['Type']
        R = Qtypes[ids]['R']
        q_ent = Qtypes[ids]['Subject']
        Questions, ans_idx = [], []
        wh_word = question.split()[0] if config.relation_span and question.split()[0].lower() in ['when','where'] else None
        
        idx = 0
        for e in q_ent:
            for i in range(len(R)):
                if i==0:
                    s, hop = e, False
                else:
                    s, hop = '{}_{}'.format(ids,idx-1), True
                if R[i] in ['Birth-Death','longer']:
                    r = ['date of birth', 'date of death']
                else:
                    r = [R[i]]
                for rx in r:
                    cur_id = '{}_{}'.format(ids,idx)
                    Questions.append({'id':cur_id, 'source':s, 'ishop':hop, 'relation':rx+" "+wh_word if wh_word else rx})
                    if i==len(R)-1:
                        ans_idx.append(cur_id)
                    idx += 1
        
        sps, evds = [], []
        for e in Questions:
            source = e['source']
            if e['ishop']:
                candidates = []
                for t in titles: 
                    s1,s2 = sim_score(all_answer[e['source']], t)
                    f1 = 2*s1*s2/(s1+s2) if s1*s2>0 else 0.0
                    candidates.append((t,f1))
                candidates.sort(key=lambda x:(-x[1],-sort_map[x[0]]))
                source = candidates[0][0]
                all_answer[e['source']] = source
                e['source'] = source
            tokens = ["[CLS]"] + tokenizer.tokenize(pro_sent(source + ' , ' + e['relation']))+["[SEP]"]
            q_len = len(tokens)
            tokens += examples[ids]['context'][sort_map[source]]
            if len(tokens)>config.max_len-1:
                tokens = tokens[:config.max_len-1]
            tokens.append("[SEP]")
            c_len = len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # BERT input
            context_idxs = torch.LongTensor(1, c_len)
            context_mask = torch.LongTensor(1, c_len)
            segment_idxs = torch.LongTensor(1, c_len)
            
            for mapping in [context_mask, segment_idxs]:
                mapping.zero_()
            
            context_idxs[0].copy_(torch.Tensor(input_ids))
            context_mask[0][:c_len] = 1
            segment_idxs[0][q_len:c_len] = 1
            
            start, end = reader(context_idxs.cuda(), segment_idxs.cuda(), context_mask.cuda())
            r1, r2 = np.argmax(start.data.cpu().numpy(), 1), np.argmax(end.data.cpu().numpy(), 1)
            
            if r1[0]>r2[0] or r1[0]<q_len:
                all_answer[e['id']] = ""
                sps.append((source, 0))
                continue
            y1,y2 = r1[0]-q_len, r2[0]-q_len
            sent_spans = examples[ids]['sent_spans'][sort_map[source]]
            ori_sents = examples[ids]['ori_sents'][sort_map[source]]
            
            text = " ".join(examples[ids]["context"][sort_map[source]][y1:y2+1]).replace(" ##","").replace("##","").split()
            y1o, y2o = 1,1
            for i in range(1,len(sent_spans)):
                if y1>=sent_spans[i][0] and y1<sent_spans[i][1]:
                    y1o = i
                if y2>=sent_spans[i][0] and y2<sent_spans[i][1]:
                    y2o = i
            ori_text = " ".join(ori_sents[y1o:y2o+1]).split()
            
            sps.append((source, y1o-1))
            all_answer[e['id']] = get_best_answer(text, ori_text)
        
        for e in Questions:
            evds.append([e['source'],e['relation'],all_answer[e['id']]])
        sp_dict[ids] = sps
        evidence_dict[ids] = evds
        
        if Type<2:
            answer_dict[ids] = all_answer[ans_idx[0]]
            continue
        
        tokens = ["[CLS]"]+comp_tokenizer.tokenize(question)+["[SEP]"]
        q_len = len(tokens)
        tokens += comp_tokenizer.tokenize(all_answer[ans_idx[0]]+' ; '+all_answer[ans_idx[1]])
        if len(tokens)>config.max_len-1:
            tokens = tokens[:config.max_len-1]
        tokens.append("[SEP]")
        c_len = len(tokens)
        input_ids = comp_tokenizer.convert_tokens_to_ids(tokens) 
        
        # BERT input
        context_idxs = torch.LongTensor(1, c_len)
        context_mask = torch.LongTensor(1, c_len)
        segment_idxs = torch.LongTensor(1, c_len)
        
        for mapping in [context_mask, segment_idxs]:
            mapping.zero_()
        
        context_idxs[0].copy_(torch.Tensor(input_ids))
        context_mask[0][:c_len] = 1
        segment_idxs[0][q_len:c_len] = 1
        
        logits = compartor(context_idxs.cuda(), segment_idxs.cuda(), context_mask.cuda())
        result = np.argmax(logits.data.cpu().numpy(), 1)
        if result[0] < 2:
            answer_dict[ids] = 'yes' if result[0]==0 else 'no'
        else:
            answer_dict[ids] = q_ent[0] if result[0]==2 else q_ent[1]
        
    prediction = {'answer': answer_dict, 'sp': sp_dict, 'evidence': evidence_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)
        

def set_config():
    parser = argparse.ArgumentParser()                                                                                                                                                                     
                                                                                                                                                                                                           
    # Required parameters
    parser.add_argument("--name", type=str, default='SingHop')
    parser.add_argument("--mode", type=str, default='dev')
    parser.add_argument("--type_pred", type=str, default='COMP/predict/pred_epoch_15.json')
    parser.add_argument("--sing_model", type=str, default='SingHop/model/ckpt_epoch_17.pth')
    parser.add_argument("--compartor_model", type=str, default='Compartor/model/ckpt_epoch_11.pth')
    parser.add_argument("--bert_model", type=str, default='/home/bert/model.en',
                        help='Currently only support bert-base-uncased and bert-large-uncased')
    
    # feature parameters
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument("--relation_span", action='store_true')
    parser.add_argument("--device", default='0', type=str, help="device to place model.")
    args = parser.parse_args()
    
    args.data_file = 'dataset/{}.json'.format(args.mode)
    args.example_file = 'features/examples_{}_singhop.json'.format(args.mode)
    
    args.prediction_path = os.path.join(args.name, 'Final')
    os.makedirs(args.prediction_path, exist_ok=True) 
    args.prediction_file = os.path.join(args.prediction_path, '{}_final.json'.format(args.mode))
    
    MODEL_dict = {"":"bert", "_bl":"bert", "_ab":"albert", "_abl":"albert", "_abxl":"albert",
                  "_rb":"roberta", "_rbl":"roberta", "_rbxl":"roberta"}
    MODEL_FILEs = {"":"../bert/model.en",  "_bl":"../bert/large.en", 
                  "_abl":"../albert/large.en", "_abxl":"../albert/xlarge.en",
                  "_rb":"../roberta/roberta.en", "_rbl":"../roberta/roberta.en"}
    args.model_type = MODEL_dict.get(args.mode,'bert')
    args.reader_model = MODEL_FILEs.get(args.mode,'../bert/model.en')
    
    return args

                                                                                                                                                                                                          
if __name__ == "__main__":
    args = set_config()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.reader_model)
    comp_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    with open(args.data_file, 'r', encoding = 'utf-8') as fh:
        questions = [{'_id':article['_id'], 'question':article['question']} for article in json.load(fh)]
    with open(args.example_file, 'r', encoding = 'utf-8') as fh:
        examples = json.load(fh)
    with open(args.type_pred, 'r', encoding = 'utf-8') as fh:
        Qtypes = json.load(fh)
        
    # Prepare Model
    reader = AlbertForQuestionAnswering.from_pretrained(args.reader_model)                                                                                                                            
    reader.cuda()
    reader.load_state_dict(torch.load(args.sing_model))
    compartor = BertForCompartor.from_pretrained(args.bert_model)
    compartor.cuda()
    compartor.load_state_dict(torch.load(args.compartor_model))
    
    Predict(model,compartor, examples, questions, Qtypes, args.prediction_file, args)