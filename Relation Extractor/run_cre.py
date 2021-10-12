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
                                                                                                                                                                                                           
                                                                                                                                                                                                           
class DataIterator(object):
    def __init__(self, example_file, config, sequential=False, subset=False):
        with open(example_file, 'r', encoding='utf-8') as fh:
            self.examples = json.load(fh)
        self.bsz = config.batch_size                                                                                                                                                                       
        self.sequential = sequential                                                                                                                                                                       
        self.max_len = config.max_len                                                                                                                                                                      
        self.example_ptr = 0
        if subset:
            self.examples = random.sample(self.examples, 32000)
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
                                                                                                                                                                                                           
        # Label tensor
        y = torch.LongTensor(self.bsz)
        y_r1 = torch.LongTensor(self.bsz)
        y_r2 = torch.LongTensor(self.bsz)
                                                                                                                                                                                                           
        while True:
            if self.example_ptr >= len(self.examples):
                break
            start_id = self.example_ptr                                                                                                                                                                    
            cur_bsz = min(self.bsz, len(self.examples) - start_id)
            cur_batch = self.examples[start_id: start_id + cur_bsz]                                                                                                                                        
            cur_batch.sort(key=lambda x: len(x["question"]), reverse=True)
                                                                                                                                                                                                           
            ids = []                                                                                                                                                                                       
            for mapping in [context_mask]:
                mapping.zero_()
            
            max_c_len = 0
            for k in range(len(cur_batch)):
                ids.append(cur_batch[k]["_id"])
                                                                                                                                                                                                           
                query_tokens = ["[CLS]"] + tokenizer.tokenize(cur_batch[k]["question"])
                if len(query_tokens) > self.max_len - 1:
                    query_tokens = query_tokens[:self.max_len - 1]
                query_tokens.append("[SEP]")
                                                                                                                                                                                                           
                q_len = len(query_tokens)
                max_c_len = max([max_c_len, q_len])
                input_ids = tokenizer.convert_tokens_to_ids(query_tokens)+ [0]*(self.max_len - q_len)
                                                                                                                                                                                                           
                context_idxs[k].copy_(torch.Tensor(input_ids))                                                                                                                                             
                context_mask[k][:q_len] = 1
                
                y[k] = Q_type.get(cur_batch[k]["type"],0)
                                                                                                                                                                                                           
                relations = []
                for e in cur_batch[k]["evidences"]:
                    if "comparison" not in cur_batch[k]['type'] or e[1] not in relations:
                        relations.append(e[1])
                if 'date of birth' in relations and 'date of death' in relations:
                    relations = relations[:-2]+['Birth-Death']
                if cur_batch[k]['type']=="comparison":
                    relations = relations+relations                                                                                                                                                        
                
                if len(relations)==2:                                                                                                                                                                                          
                    y_r1[k] = r_dict.get(relations[0],34)
                    y_r2[k] = r_dict.get(relations[1],34)
                else:
                    y_r1[k],y_r2[k] = 0,0
                                                                                                                                                                                                           
            self.example_ptr += cur_bsz                                                                                                                                                                    
                                                                                                                                                                                                           
            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'y': y[:cur_bsz],
                'y1': y_r1[:cur_bsz],
                'y2': y_r2[:cur_bsz],
                'ids': ids}
                                                                                                                                                                                                         
                                                                                                                                                                                                           
def predict(model, dataloader, writer, prediction_file):
    model.eval()                                                                                                                                                                                           
    answer_dict = {}
    t_acc, r_acc = [],[]
    dataloader.refresh()                                                                                                                                                                                   
    for batch in tqdm(dataloader):
        batch = {k:v.cuda() if k not in ['ids'] else v for k,v in batch.items()}
        t_logits, r1, r2 = model(batch)
                                                                                                                                                                                                           
        t,r1s,r2s = np.argmax(t_logits.data.cpu().numpy(), 1), r1.data.cpu().numpy(), r2.data.cpu().numpy()
                                                                                                                                                                                                           
        for i in range(t.shape[0]):                                                                                                                                                                        

            cur_id = batch['ids'][i]
                                                                                                                                                                                                           
            t_acc.append(1 if t[i]==batch['y'][i] else 0)
                                                                                                                                                                                                           
            if int(t[i]) == 2:
                x = int(np.argmax(r1s[i]+r2s[i]))
                r_acc.append(1 if x==batch['y1'][i] else 0)
                r_acc.append(1 if x==batch['y2'][i] else 0)
                answer_dict[cur_id] = {'Type':int(t[i]),'R':[relations[x]]}
            else:
                x1,x2 = int(np.argmax(r1s[i])), int(np.argmax(r2s[i]))
                r_acc.append(1 if x1==batch['y1'][i] else 0)
                r_acc.append(1 if x2==batch['y2'][i] else 0)
                answer_dict[cur_id] = {'Type':int(t[i]),'R':[relations[x1],relations[x2]]}
    writer.write('T-acc: {:.3f}\t R_acc:{:.3f}\n'.format(sum(t_acc)/len(t_acc),sum(r_acc)/len(r_acc)))
    with open(prediction_file, 'w') as f:
        json.dump(answer_dict, f)                                                                                                                                                                          
                                                                                                                                                                                                         
    model.train()                                                                                                                                                                                          
                                                                                                                                                                                                           
                                                                                                                                                                                                           
def set_config():
    parser = argparse.ArgumentParser()                                                                                                                                                                     
                                                                                                                                                                                                           
    # Required parameters
    parser.add_argument("--name", type=str, default='CRE')
    parser.add_argument("--train_file", type=str, default='dataset/train.json')
    parser.add_argument("--dev_file", type=str, default='dataset/dev.json')
    parser.add_argument("--test_file", type=str, default='dataset/test.json')
    parser.add_argument("--bert_model", type=str, default='../bert/model.en',
                        help='Currently only support bert-base-uncased and bert-large-uncased')
                                                                                                                                                                                                           
    # feature parameters
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--qat_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_bert_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
                                                                                                                                                                                                           
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    parser.add_argument("--do_continue", type=int, default=-1, help="Whether to run eval.")
    parser.add_argument('--eval_list', type=int, default=0)
                                                                                                                                                                                                           
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
    test_dataset = DataIterator(args.test_file, args, sequential=True)
                                                                                                                                                                                                           
    # Prepare Model
    model = BertForCompMhop.from_pretrained(args.bert_model)                                                                                                                                               
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model = model.module                                                                                                                                                                               
    model.cuda()
                                                                                                                                                                                                           
    # Prepare Optimizer
    lr = args.lr                                                                                                                                                                                           
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)                                                                                                                                           
    t_total = int((170000*args.epochs - 138000*args.qat_epochs)/args.batch_size/args.gradient_accumulation_steps)
                                                                                                                                                                                                           
    global_step = 1
    if not args.do_eval:
        start = 0
        if args.do_continue >= 0:
            start = args.do_continue + 1
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(start-1))))
            global_step = 32000*start if start <= args.qat_epochs else 170000*start - 138000*args.qat_epochs
            global_step = int(global_step/args.batch_size/args.gradient_accumulation_steps)
                                                                                                                                                                                                           
        model.train()
        for epc in trange(start, int(args.epochs), desc="Epoch"):
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
            predict(model, eval_dataset, writer,                                                                                                                                                           
                os.path.join(args.prediction_path, 'pred_epoch_{}.json'.format(epc)))
    elif args.eval_list <= 0:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "ckpt_epoch_15.pth")))
        predict(model, test_dataset, writer, os.path.join(args.prediction_path, 'pred_test.json'))
    else:
        for epc in range(args.eval_list):
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc))))
            predict(model, eval_dataset, writer,                                                                                                                                                           
                os.path.join(args.prediction_path, 'pred_epoch_{}.json'.format(epc)))