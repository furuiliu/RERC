from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import logging
import argparse
from tqdm import tqdm, trange
from numpy.random import shuffle
                                                                                                                                                                                                           
import numpy as np
import random
import ujson as json
                                                                                                                                                                                                           
from func import *
                                                                                                                                                                                                           
                                                                                                                                                                                                           
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                                       
month = {'apr': 4,
 'april': 4,
 'aug': 8,
 'august': 8,
 'dec': 12,
 'december': 12,
 'feb': 2,
 'february': 2,
 'jan': 1,
 'january': 1,
 'jul': 7,
 'july': 7,
 'jun': 6,
 'june': 6,
 'mar': 3,
 'march': 3,
 'may': 5,
 'nov': 11,
 'november': 11,
 'oct': 10,
 'october': 10,
 'sep': 9,
 'september': 9}

Month = [item for item in month.items()]+[(k.capitalize(),month[k]) for k in month.keys()]
R = [('older',0),('first',0),('earlier',0),('younger',1),('recently',1),('lastly',1),('longer',1)]                                                                                                                                                                                                          
def Out(y,m,d,t):
    if t==0:
        return " ".join(list(str(y))), y*1000
    if t==1:
        return " ".join([m[0]]+list(str(y))), y*1000+m[1]*50
    if t==2:
        return " ".join([m[0]]+list(str(d))), m[1]*50+d
    if t==3:
        return " ".join(list(str(d))+[m[0]]+list(str(y))), y*1000+m[1]*50+d
    return " ".join([m[0]]+list(str(d)+','+str(y))), y*1000+m[1]*50+d


class NumIterator(object):
    def __init__(self, size, config):
        self.examples = []
        Y = np.random.choice(np.arange(1000,2022), 2*size)
        M = np.random.choice(np.arange(len(Month)), 2*size)
        D = np.random.choice(np.arange(1,32), 2*size)
        T = np.random.choice(np.arange(5), 2*size)

        for i in range(size):
            s,v = Out(Y[i],Month[M[i]],D[i],T[i])
            s1,v1 = Out(Y[i+size],Month[M[i+size]],D[i+size],T[i])
            s2,v2 = Out(Y[i+size],Month[M[i+size]],D[i+size],T[i+size])
            s3,v3 = Out(Y[i],Month[M[i]],D[i],3+i%2)
            s4,v4 = Out(Y[i],Month[M[i+size]],D[i+size],3+i%3)
            if v!=v1:
                flag = 0 if v<v1 else 1
                self.examples.append({'question':R[i%7][0],'num1':s,'num2':s1,'label':(R[i%7][1]+flag)%2})
            if T[i]==2 or T[i+size] == 2:
                if v3!=v4:
                    flag = 0 if v3<v4 else 1
                    self.examples.append({'question':R[i%7][0],'num1':s3,'num2':s4,'label':(R[i%7][1]+flag)%2})
            else:
                if v!=v2 and T[i]!=T[i+size]:
                    flag = 0 if v<v2 else 1
                    self.examples.append({'question':R[i%7][0],'num1':s,'num2':s2,'label':(R[i%7][1]+flag)%2})
        shuffle(self.examples)
        self.bsz = config.batch_size                                                                 
        self.max_len = config.max_len        
        self.example_ptr = 0
    
    def refresh(self):
        self.example_ptr = 0
        shuffle(self.examples)
                                                                                                                                                                                                           
    def empty(self):
        return self.example_ptr >= len(self.examples)
                                                                                                                                                                                                           
    def __len__(self):
        return int(np.ceil(len(self.examples)/self.bsz))
                                                                                                                                                                                                           
    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, self.max_len)                                                                                                                                            
        context_mask = torch.LongTensor(self.bsz, self.max_len)                                                                                                                                            
        segment_idxs = torch.LongTensor(self.bsz, self.max_len)                                                                                                                                                                                                   
        # Label tensor
        y = torch.LongTensor(self.bsz)
                                                                                                                                                                                                           
        while True:
            if self.example_ptr >= len(self.examples):
                break
            start_id = self.example_ptr                                                                                                                                                                    
            cur_bsz = min(self.bsz, len(self.examples) - start_id)
            cur_batch = self.examples[start_id: start_id + cur_bsz]                                                                                                                                        
            for mapping in [context_mask, segment_idxs]:
                mapping.zero_()
            
            max_c_len = 0
            for k in range(len(cur_batch)):
            
                query_tokens = ["[CLS]"] + tokenizer.tokenize(cur_batch[k]["question"])+["[SEP]"]
                q_len = len(query_tokens)
                query_tokens += tokenizer.tokenize(cur_batch[k]["num1"])+["[SEP]"]
                n_len = len(query_tokens)
                query_tokens += tokenizer.tokenize(cur_batch[k]["num2"])+["[SEP]"]
                c_len = len(query_tokens)
                max_c_len = max([max_c_len, c_len])
                input_ids = tokenizer.convert_tokens_to_ids(query_tokens)+ [0]*(self.max_len - c_len)
                
                context_idxs[k].copy_(torch.Tensor(input_ids))                
                context_mask[k][:c_len] = 1
                segment_idxs[k][q_len:c_len] = 1
                
                y[k] = cur_batch[k]["label"]

            self.example_ptr += cur_bsz
            
            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'y': y[:cur_bsz],
                'ids': ids} 
                
class DataIterator(object):
    def __init__(self, example_file, config, sequential=False, subset=False):
        with open(example_file, 'r', encoding='utf-8') as fh:
            self.examples = json.load(fh)
        self.sep = config.sep
        self.keyword = config.keyword
        self.bsz = config.batch_size                                                                                                                                                                       
        self.sequential = sequential                                                                                                                                                                       
        self.max_len = config.max_len                                                                                                                                                                      
        self.example_ptr = 0
        if subset:
            self.examples = random.sample(self.examples, int(0.2*len(self.examples)))
        elif not sequential:
            shuffle(self.examples)
                                                                                                                                                                                                           
    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.examples)
                                                                                                                                                                                                           
    def empty(self):
        return self.example_ptr >= len(self.examples)
                                                                                                                                                                                                           
    def __len__(self):
        return int(np.ceil(len(self.examples)/self.bsz))
                                                                                                                                                                                                           
    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, self.max_len)                                                                                                                                            
        context_mask = torch.LongTensor(self.bsz, self.max_len)                                                                                                                                            
        segment_idxs = torch.LongTensor(self.bsz, self.max_len)                                                                                                                                                                                                   
        # Label tensor
        y = torch.LongTensor(self.bsz)
                                                                                                                                                                                                           
        while True:
            if self.example_ptr >= len(self.examples):
                break
            start_id = self.example_ptr                                                                                                                                                                    
            cur_bsz = min(self.bsz, len(self.examples) - start_id)
            cur_batch = self.examples[start_id: start_id + cur_bsz]                                                                                                                                        
            cur_batch.sort(key=lambda x: len(x["question"]), reverse=True)
                                                                                                                                                                                                           
            ids = []                                                                                                                                                                                       
            for mapping in [context_mask, segment_idxs]:
                mapping.zero_()
            
            max_c_len = 0
            for k in range(len(cur_batch)):
                ids.append(cur_batch[k]["_id"])
                
                # query_tokens = ["[CLS]"] + tokenizer.tokenize(cur_batch[k]["Num1"])+["[SEP]"]
                # q_len = len(query_tokens)
                # query_tokens += tokenizer.tokenize(cur_batch[k]["Num2"])
                # if len(query_tokens) > self.max_len - 1:
                    # query_tokens = query_tokens[:self.max_len - 1]
                # query_tokens.append("[SEP]") 
                
                # c_len = len(query_tokens)
                # max_c_len = max([max_c_len, c_len])
                # input_ids = tokenizer.convert_tokens_to_ids(query_tokens)+ [0]*(self.max_len - c_len)
                
                # context_idxs[k].copy_(torch.Tensor(input_ids))                
                # context_mask[k][:c_len] = 1
                # segment_idxs[k][q_len:c_len] = 1
                
                # if cur_batch[k]["answer"] == 'yes':
                    # y[k] = 0
                # elif cur_batch[k]["answer"] == 'no':
                    # y[k] = 1 if len(cur_batch[k]["Num1"])>len(cur_batch[k]["Num2"]) else 2
                # else:
                    # flag = 0
                    # for w in ['older','first','earlier']:
                        # if w in question:
                            # flag = 1
                            # break
                    # if IsSame(cur_batch[k]["answer"],cur_batch[k]["Subject"][0]):
                        # y[k] = 1 + flag
                    # else:
                        # y[k] = 2 - flag
                
                flag = 0
                if self.keyword:
                    for w in ['the same','older','first','earlier','recently','younger','later','longer','more']:
                        if w in cur_batch[k]["question"]:
                            query_tokens = ["[CLS]"] + tokenizer.tokenize(w)+["[SEP]"]
                            flag = 1
                            break
                if flag == 0:
                    query_tokens = ["[CLS]"] + tokenizer.tokenize(cur_batch[k]["question"])+["[SEP]"]                                                                                                                                                                                       
                q_len = len(query_tokens)
                if self.sep:
                    query_tokens += tokenizer.tokenize(cur_batch[k]["Num1"])+["[SEP]"]+tokenizer.tokenize(cur_batch[k]["Num2"])
                else:
                    query_tokens += tokenizer.tokenize(cur_batch[k]["Num1"]+' ; '+cur_batch[k]["Num2"])
                
                if len(query_tokens) > self.max_len - 1:
                    query_tokens = query_tokens[:self.max_len - 1]
                query_tokens.append("[SEP]")                                                                                                                                                                                          
                c_len = len(query_tokens)
                
                max_c_len = max([max_c_len, c_len])
                input_ids = tokenizer.convert_tokens_to_ids(query_tokens)+ [0]*(self.max_len - c_len)
                                                                                                                                                                                                           
                context_idxs[k].copy_(torch.Tensor(input_ids))                                                                                                                                             
                context_mask[k][:c_len] = 1
                segment_idxs[k][q_len:c_len] = 1
                
                if cur_batch[k]["answer"] == 'yes':
                    y[k] = 0
                elif cur_batch[k]["answer"] == 'no':
                    y[k] = 1
                elif cur_batch[k]["answer"] == cur_batch[k]["Num1"]:
                    y[k] = 2
                else:
                    y[k] = 3
                
                                                                                                                                                                                                           
            self.example_ptr += cur_bsz                                                                                                                                                                    
                                                                                                                                                                                                           
            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'y': y[:cur_bsz],
                'ids': ids}
                                                                                                                                                                                                         
                                                                                                                                                                                                           
def eval(model, dataloader, writer):
    model.eval()
    n = 4 
    mat = [[0]*n for _ in range(n)]
    dataloader.refresh()
    
    for batch in tqdm(dataloader):
        batch = {k:v.cuda() if k not in ['ids'] else v for k,v in batch.items()}
        logits = model(batch)
        t = np.argmax(logits.data.cpu().numpy(), 1)
        
        for i in range(t.shape[0]):        
            cur_id = batch['ids'][i]
            mat[batch['y'][i]][t[i]] += 1
                                                                                                                                                                                                           
    per_num = [sum(row) for row in mat]
    total = sum(per_num)
    
    writer.write('Acc: {:.3f}\n'.format(sum([mat[i][i] for i in range(n)])/total))
    writer.write('PerAcc: {}\n'.format('\t'.join(['{}:{:.3f}'.format(i,mat[i][i]/per_num[i]) for i in range(n)])))
    writer.write('\n'.join(['\t'.join([str(a) for a in row]) for row in mat])+'\n')
                                                                                                                                                                                                         
    model.train()                                                                                                                                                                                          
                                                                                                                                                                                                           
                                                                                                                                                                                                           
def set_config():
    parser = argparse.ArgumentParser()                                                                                                                                                                     
                                                                                                                                                                                                           
    # Required parameters
    parser.add_argument("--name", type=str, default='Comparison')
    parser.add_argument("--train_file", type=str, default='dataset/train_compare.json')
    parser.add_argument("--dev_file", type=str, default='dataset/dev_compare.json')
    parser.add_argument("--test_file", type=str, default='dataset/test.json')
    parser.add_argument("--bert_model", type=str, default='../bert/model.en',
                        help='Currently only support bert-base-uncased and bert-large-uncased')
                                                                                                                                                                                                           
    # feature parameters
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--qat_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_bert_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
                                                                                                                                                                                                           
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    parser.add_argument("--do_continue", type=int, default=-1, help="Whether to run eval.")
    parser.add_argument('--eval_list', type=str, default='0')
    parser.add_argument("--sep", action='store_true')
    parser.add_argument("--keyword", action='store_true')
    parser.add_argument("--data_size", type=int, default=20000)
                                                                                                                                                                                                           
    # device
    parser.add_argument("--device", default='0', type=str, help="device to place model.")
                                                                                                                                                                                               
    args = parser.parse_args()
                                                                                                                                                                                                           
    args.checkpoint_path = os.path.join(args.name, 'model')
    args.prediction_path = os.path.join(args.name, 'predict')
    args.best_model = os.path.join(args.checkpoint_path, "best_model.pth")
                                                                                                                                                                                                           
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.prediction_path, exist_ok=True)
                                                                                                                                                                                                    
    return args
                                                                                                                                                                                                           
                                                                                                                                                                                                           
if __name__ == "__main__":
    args = set_config()                                                                                                                                                                                    
    writer = open(args.name + '/train.log', 'w', encoding = 'utf-8',buffering = 1)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)                                                                                                                                             
                                                                                                                                                                                                           
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
                                                                                                                                                                                                           
    # Set GPU Issue
    n_gpu = torch.cuda.device_count()                                                                                                                                                                      
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)
                                                                                                                                                                                                           
    logger.info("n_gpu: {} Grad_Accum_steps: {} Batch_size: {}".format(
                n_gpu, args.gradient_accumulation_steps, args.batch_size))                                                                                                                                 
                                                                                                                                                                                                           
    # Set Seeds
    random.seed(args.seed)                                                                                                                                                                                 
    np.random.seed(args.seed)                                                                                                                                                                              
    torch.manual_seed(args.seed)                                                                                                                                                                           
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)                                                                                                                                                              
                                                                                                                                                                                                           
    # Set datasets
    Full_Loader = DataIterator(args.train_file, args)
    Subset_Loader = DataIterator(args.train_file, args, subset=True)
    eval_dataset = DataIterator(args.dev_file, args, sequential=True)
    
    
    # Full_Loader = NumIterator(args.data_size, args)
    # Subset_Loader = NumIterator(int(args.data_size*0.2), args)
    # eval_dataset = NumIterator(int(args.data_size*0.1), args)
    
    # Prepare Model
    model = BertForCompare.from_pretrained(args.bert_model)                                                                                                                                               
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model = model.module                                                                                                                                                                               
    model.cuda()
                                                                                                                                                                                                           
    # Prepare Optimizer
    lr = args.lr                                                                                                                                                                                           
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)                                                                                                                                           
    t_total = int(len(Full_Loader)*args.epochs/args.batch_size/args.gradient_accumulation_steps)
                                                                                                                                                                                                           
    global_step = 1
    if not args.do_eval:
                                                                                                                                                                                                           
        model.train()
        for epc in trange(int(args.epochs), desc="Epoch"):
            tr_loss = 0
                                                                                                                                                                                                           
            if epc <= args.qat_epochs:
                Loader = Subset_Loader                                                                                                                                                                     
            else:
                Loader = Full_Loader                                                                                                                                                                       
            Loader.refresh()                                                                                                                                                                               
                                                                                                                                                                                                           
            for step, batch in enumerate(tqdm(Loader, desc="Iteration")):
                batch = {k:v.cuda() if k not in ['ids'] else v for k,v in batch.items()}
                loss = model(batch, is_train = True)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps                                                                                                                                         
                loss.backward()                                                                                                                                                                            
                                                                                                                                                                                                           
                tr_loss += loss.item()                                                                                                                                                                     
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.lr * warmup_linear(global_step / t_total, args.warmup_proportion)                                                                                                  
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()                                                                                                                                                                       
                    optimizer.zero_grad()                                                                                                                                                                  
                    global_step += 1
                                                                                                                                                                                                           
                if global_step % 1500 == 0:
                    add_figure(args.name, writer, global_step, tr_loss/step)                                                                                                                               
                                                                                                                                                                                                           
            # Save a trained model
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc)))
            writer.write('Epoch {} Predict: \n'.format(epc))
            eval(model, eval_dataset, writer)
    else:
        for epc in args.eval_list.split(','):
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc))))
            writer.write('Epoch {} Predict: \n'.format(epc))
            eval(model, eval_dataset, writer)