import random
import ujson as json
import argparse
from collections import Counter
from tqdm import tqdm
import re
import os
from transformers import (BertTokenizer, RobertaTokenizer, AlbertTokenizer)

MODEL_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
    'albert': AlbertTokenizer,
}

with open('dataset/id2ent.json','r',encoding = 'utf-8') as fr:
    id2ent = json.load(fr)
with open('dataset/ent2id.json','r',encoding = 'utf-8') as fr:
    ent2id = json.load(fr)

def pro_sent(sent):
    sent = re.sub('[^a-z0-9,.:?!|# ]','',sent.lower().replace('-',' '))
    sent = re.sub('[,:?!.]',' \1', sent+' ')
    return ' '.join(sent.split())

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'./stanford-corenlp-4.2.1')
for mode in ['dev','train','test']:
    if not os.path.exists('entities_{}.json'.format(mode)):
        out = {}
        with open('dataset/{}.json'.format(mode),'r') as fh:
            for article in json.load(fh):
                context = [[[w[0] for w in nlp.ner(pro_sent(s)) if w[1] != 'O'] for s in c[1]] for c in article["context"]]
                question = [w[0] for w in nlp.ner(pro_sent(article["question"])) if w[1] != 'O']
                out[article["_id"]] = {'context':context, 'question':question}
        with open('entities_{}.json'.format(mode),'w') as fw:
            json.dump(out,fw,indent=2)
nlp.close()       

def lcs(a,b):
	lena=len(a)
	lenb=len(b)
	c=[[0 for i in range(lenb+1)] for j in range(lena+1)]
	for i in range(lena):
		for j in range(lenb):
			if a[i]==b[j]:
				c[i+1][j+1]=c[i][j]+1
			elif c[i+1][j]>c[i][j+1]:
				c[i+1][j+1]=c[i+1][j]
			else:
				c[i+1][j+1]=c[i][j+1]
	return c[-1][-1]

def IsSameE(e1,e2):
    if e1 in ent2id and e2 in ent2id:
        return ent2id[e1]==ent2id[e2]
    E1,E2 = [e1], [e2]
    if e1 in ent2id:
        E1 = id2ent[ent2id[e1]]
    if e2 in ent2id:
        E2 = id2ent[ent2id[e2]]
    
    for ent1 in E1:
        for ent2 in E2:
            if lcs(ent1,ent2)/min(len(ent1),len(ent2)) >= 0.8:
                return True
    return False
  

def sim_score(a,b):
    question_tokens = tokenizer.tokenize(pro_sent(a))
    para = tokenizer.tokenize(pro_sent(b))
    common_with_question = Counter(para) & Counter(question_tokens)
    correct_preds = sum(common_with_question.values())
    if correct_preds==0:
        return 0.0,0.0
    return float(correct_preds) / len(question_tokens), float(correct_preds) / len(para)


def find_entity(sents, entities, is_sp = []):
    out_sents, count, idxs = [],[], []
    for idx, sent in enumerate(sents):
        for title, es in entities:
            c = 0
            for e in es:
                if e in sent:
                    sent = sent.replace(e,title)
                    c += 1
            count.append(c)
        out_sents.append(sent)
        idxs.append((idx,idx in is_sp))
            
    return out_sents,count,idxs
   
    

def _process_article(article, entities, tokenizer, is_eval = False):
    context = [[pro_sent(s) for s in [c[0]]+c[1]] for c in article["context"]]
    n = len(context)
    
    # 
    is_sp = {}
    answer = []
    if not is_eval:
        for s in article["supporting_facts"]:
            is_sp[s[0]] = is_sp.get(s[0],[]) + [s[1]]
        for e in article["evidences"]:
            answer.append(' '.join(tokenizer.tokenize(pro_sent(e[2]))))
        
    
    # find entities
    paras, e_mat, the_idx = [],[],[]
    for i in range(n):
        out_sents,count,idxs = find_entity(context[i], entities, is_sp.get(article["context"][i][0],[]))
        paras.append(out_sents)
        e_mat.append(count)
        the_idx.append(idxs)
    
    # create entities tree
    titles = [c[0] for c in article["context"]]
    title_dict = {titles[i]:i for i in range(n)}
    
    EntityList = entities['question']   
    state = [0]*n
    grade, flag = 0,1
    while flag:
        flag = 0
        l = len(EntityList)
        for i in range(n):
            if state[i]>=0:
                continue
            for j in range(len(entities['context'][i])):
                sim,_ = sim_score(article['question'],article['context'][i][j])
                if sim < 0.65:
                    continue
                new_E = []
                for e in entities['context'][i][j]:
                    flag_e = 0
                    for k in range(l):
                        if IsSameE(e, EntityList[k]):
                            flag_e = 1
                            break
                    if flag_e==0:
                        new_E.append(e)
                if len(new_E)<len(entities['context'][i][j]) and len(new_E)>0:
                    EntityList += new_E
                    flag = 1
                    if state[i]<0:
                        state[i] = grade + 1
        EntityList = EntityList[l:]     
        grade += 1
    state = [0.0 if state[i] < 0 else 1.0-0.5*state[i]/grade for i in range(n)]
     
    # 
    if not is_eval:
        for s in article["supporting_facts"]:
            idx = title_dict[s[0]]
            state[idx] += 10.0
    
    score = [sum(v) for v in e_mat]
    total = sum(score)
    state = [(i,state[i], score[i]/total) for i in range(n)]               
    state.sort(key = lambda x:(-x[1]-x[2],-x[1],x[0]))
    
    # 
    sort_map={}
    context_idxs = []
    ori_sents = []
    sent_spans = []
    ans_spans = [[0,0,0] for _ in range(len(answer))]
    for i in range(n):
        idx = state[i][0]
        sort_map[titles[idx]] = i
        input_ids = []
        ori_text = []
        sents = []
        for j in range(len(paras[idx])):
            start = len(input_ids)
            input_ids += tokenizer.tokenize(paras[idx][j])
            sents.append([start,len(input_ids)])
            ori_text.append(article["context"][idx][1][the_idx[idx][j][0]-1] if the_idx[idx][j][0]>0 else titles[idx])
        context_idxs.append(input_ids)
        ori_sents.append(ori_text)
        sent_spans.append(sents)
    if not is_eval:
        for p in range(len(answer)):
            candidates = []
            for t in titles: 
                s1,s2 = sim_score(article["evidences"][p][0], t)
                candidates.append((t,s1,s2))
            candidates.sort(key=lambda x:(-x[1],-x[2]))
            i = sort_map[candidates[0][0]]
            y1,y2 = -1,-1
            input_str = ' '.join(context_idxs[i])
            input_str = input_str.replace(answer[p],'{'+answer[p]+'}')
            input_mask = input_str.split()
            for q in range(len(input_mask)):
                if '{' in input_mask[q]:
                    y1 = q
                if '}' in input_mask[q]:
                    y2 = q
                    break
            ans_spans[p][0],ans_spans[p][1],ans_spans[p][2] = y1,y2,i

    # 
    q_idxs = []
    q_spans = []
    if not is_eval:
        spans = []
        for i in range(len(answer)):
            if ans_spans[i][0]>ans_spans[i][1] or ans_spans[i][0]<0:
                print(article["evidences"][i])
                continue
            flag = 1
            if i < len(answer)-1:
                if article["evidences"][i][0]==article["evidences"][i+1][0] and article["evidences"][i][1]==article["evidences"][i+1][1]:
                    flag = 0
            spans.append(ans_spans[i])
            if flag>0:
                if args.span_relation:
                    flag_r = 0
                    if len(article['relation_evidence'])>i:
                        r = article['relation_evidence'][i]
                        if r in article['question']
                            if article['question'].split()[0] in ['When','Where','when','where']:
                                r = r + ' ' + article['question'].split()[0]
                            question_ids = tokenizer.tokenize(pro_sent(article["evidences"][i][0]+' , '+r))
                            flag_r = 1
                    if flag_r == 0:
                        continue
                else:
                    question_ids = tokenizer.tokenize(pro_sent(' , '.join(article["evidences"][i][:2])))
                if len(spans)>1:
                    flag = 0
                    spans.sort(key=lambda x:x[0])
                    for i in range(len(spans)-1):
                        if spans[i+1][0]-spans[i][1]>3:
                            flag=1
                            break
                    if flag>0:
                        spans = []
                        continue
                    
                    y1,y2 = spans[0][0],spans[-1][1]
                else:
                    y1,y2 = spans[0][0],spans[0][1]
                q_idxs.append(question_ids)
                q_spans.append([y1,y2,spans[0][2]])
                spans = []
    
    #  
    ids = article["_id"]
    example = {'ids': ids,
            'context': context_idxs,
            'ori_sents': ori_sents,
            'title': titles,
            'sent_spans': sent_spans}
            
    questions = []
    if not is_eval:
        for q, span in zip(q_idxs,q_spans):
            questions.append({'q_idxs': q,
                'ids': ids,
                'label': span
                })
    return example,questions        
                
 
def process_file(filename, entityfile, name, tokenizer, is_eval = False):
    if args.span_relation:
        name  = name + '_span'
    print('Processing for '+name)
    
    with open(entityfile, 'r') as fh:
        entities = json.load(fh)
    
    data = json.load(open(filename, 'r'))
    outputs = [_process_article(article, entities[article['_id']], tokenizer, is_eval) for article in tqdm(data)]
    
    examples = {e[0]['ids']:e[0] for e in outputs}

    with open('features/examples_{}.json'.format(name), 'w', encoding='utf-8') as fw:
        json.dump(examples,fw)
    
    if not is_eval:
        features = []
        for _, e in outputs:
            features += e
        random.shuffle(features)
        print("create {} questions in total".format(len(features)))
        with open('features/{}.json'.format(name), 'w', encoding='utf-8') as fw:
            json.dump(features,fw)
    
    return
    

if __name__ == "__main__":
    random.seed(13)
    
    MODEL_dict = {"":"bert", "_bl":"bert", "_ab":"albert", "_abl":"albert", "_abxl":"albert",
                  "_rb":"roberta", "_rbl":"roberta", "_rbxl":"roberta"}
    MODEL_FILEs = {"":"/home/bert/model.en",  "_bl":"/home/bert/large.en", 
                  "_abl":"/home/albert/large.en", "_abxl":"/home/albert/xlarge.en",
                  "_rb":"/home/roberta/roberta.en", "_rbl":"/home/roberta/roberta.en"}
                  
    parser = argparse.ArgumentParser()                                                                                                                                                                     
    parser.add_argument("--mode", type=str, default='')
    parser.add_argument("--span_relation", action='store_true')
    args = parser.parse_args()
    args.model_type = MODEL_dict.get(args.mode,'bert')
    args.bert_model = MODEL_FILEs.get(args.mode,'/home/bert/model.en')
    tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.bert_model) 
    
        
    for mode in ['dev','train','test']:
        filename = 'dataset/{}.json'.format(mode) if not args.span_relation else 'dataset/{}_sre.json'
        entityfile = 'entities_{}.json'.format(mode)
        process_file(filename, entityfile, mode+'_singhop'+args.mode, ent2id, id2ent, tokenizer, is_eval = (mode=='test'))
        
        


