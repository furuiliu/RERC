from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import logging
import argparse
from tqdm import tqdm, trange
from numpy.random import shuffle
                                                                                                                                                                                                           
import random
import ujson as json
                                                                                                                                                                                                           
from func import *

                                                                                                                                                                                                           
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                                       

IGNORE_INDEX = -100
                                                                                                                                                                                                    
                                                                                                                                                                                                           
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
                                                                                                                                                                                                           
                                                                                                                                                                                                           
def add_figure(name, writer, global_step, train_loss):
    writer.write(name + '_data/train_loss: {} \t global_step:{} \n'.format(train_loss, global_step))
    return
 

class DataIterator(object):
    def __init__(self, examples, source_dict, config, sequential=False):
        self.examples = examples                                                                                                                                                                           
        self.source_dict = source_dict                                                                                                                                                                     
        self.bsz = config.batch_size                                                                                                                                                                       
        self.sequential = sequential                                                                                                                                                                       
        self.max_len = config.max_len                                                                                                                                                                      
        self.all_paras = config.all_paras                                                                                                                                                                  
        self.example_ptr = 0
        if not sequential:
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
        y1 = torch.LongTensor(self.bsz)
        y2 = torch.LongTensor(self.bsz)
                                                                                                                                                                                                           
        while True:
            if self.example_ptr >= len(self.examples):
                break
            start_id = self.example_ptr                                                                                                                                                                    
            cur_bsz = min(self.bsz, len(self.examples) - start_id)
            cur_batch = self.examples[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: len(x["q_idxs"]), reverse=True)
                                                                                                                                                                                                           
            ids,q_length, source = [],[],[]
            for mapping in [context_mask, segment_idxs]:
                mapping.zero_()
                                                                                                                                                                                                           
            max_c_len = 0
            for k in range(len(cur_batch)):
                cur_id = cur_batch[k]["ids"].split('_')[0]
                ids.append(cur_batch[k]["ids"])
                p_idx = cur_batch[k]["label"][2]
                                                                                                                                                                                                           
                query_tokens = ["[CLS]"] + cur_batch[k]["q_idxs"]+["[SEP]"]
                q_len = len(query_tokens)
                query_tokens += self.source_dict[cur_id]["context"][p_idx]
                if self.all_paras:
                    for i in range(len(self.source_dict[cur_id]["context"])):
                        if len(query_tokens) >= self.max_len - 1:
                            break
                        if i==p_idx:
                            continue
                        query_tokens += self.source_dict[cur_id]["context"][i]
                                                                                                                                                                                                           
                if len(query_tokens) > self.max_len - 1:
                    query_tokens = query_tokens[:self.max_len - 1]
                query_tokens.append("[SEP]")
                                                                                                                                                                                                           
                c_len = len(query_tokens)
                max_c_len = max([max_c_len, c_len])
                input_ids = tokenizer.convert_tokens_to_ids(query_tokens)+ [0]*(self.max_len - c_len)
                                                                                                                                                                                                           
                context_idxs[k].copy_(torch.Tensor(input_ids))
                context_mask[k][:c_len] = 1
                segment_idxs[k][q_len:c_len] = 1
                                                                                                                                                                                                           
                y1[k] = cur_batch[k]["label"][0] + q_len
                y2[k] = cur_batch[k]["label"][1] + q_len
                q_length.append(q_len)
                source.append(p_idx)
                                                                                                                                                                                                           
            self.example_ptr += cur_bsz                                                                                                                                                                    
                                                                                                                                                                                                           
            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'q_len': q_length,
                'p_idx': source,
                'ids': ids}
                                                                                                                                                                                                           
def F1(a,b,c,d):
    correct = max([0,min(b,d)-max(a,c)])
    if correct == 0:
        return 0.0,0.0,0.0
    pred, recall = correct/(b-a), correct/(d-c)
    return 2*pred*recall/(pred+recall), pred, recall
                                                                                                                                                                                                           
                                                                                                                                                                                                           
def predict(model, dataloader, example_dict, writer, prediction_file):
    model.eval()                                                                                                                                                                                           
    answer_dict = {}                                                                                                                                                                                       
    f, Pred, Recall = [],[],[]                                                                                                                                                                             
    dataloader.refresh()                                                                                                                                                                                                                  
    for batch in tqdm(dataloader):
        batch = {k:v.cuda() if k not in ['ids','q_len','p_idx'] else v for k,v in batch.items()}
        input_ids, attention_mask, token_type_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        start, end = model(input_ids, token_type_ids, attention_mask)
                                                                                                                                                                                                           
        r1, r2 = np.argmax(start.data.cpu().numpy(), 1), np.argmax(end.data.cpu().numpy(), 1)
                                                                                                                                                                                                           
        for i in range(r1.shape[0]):                                                                                                                                                                       

            cur_id, idx = batch['ids'][i].split('_')[0], batch['p_idx'][i]
            q_len = int(batch['q_len'][i])
            a,b,c,d = int(r1[i])-q_len, int(r2[i])-q_len, int(batch['y1'][i])-q_len, int(batch['y2'][i])-q_len
            f1,pred,recall = F1(a,b,c,d)
            f.append(f1)
            Pred.append(pred)
            Recall.append(recall)
                                                                                                                                                                                                           
            predt = ""
            if b>=a:
                predt = ' '.join(example_dict[cur_id]['context'][int(idx)][a:b+1]).replace(' ##','')
            label = ' '.join(example_dict[cur_id]['context'][int(idx)][c:d+1]).replace(' ##','')
                                                                                                                                                                                                           
            answer_dict[batch['ids'][i]] = {'predict':(predt,a,b),'label':(label,c,d)}
                                                                                                                                                                                                           
    writer.write('F1: {:.3f}\t Predict: {:.3f}\t Recall: {:.3f}\n'.format(sum(f)/len(f),sum(Pred)/len(f), sum(Recall)/len(f)))
                                                                                                                                                                                                           
    with open(prediction_file, 'w') as f:
        json.dump(answer_dict, f, indent = 2)                                                                                                                                                              
                                                                                                                                                                                               
    model.train()                                                                                                                                                                                                                


def set_config():
    parser = argparse.ArgumentParser()                                                                                                                                                                     
                                                                                                                                                                                                           
    # Required parameters
    parser.add_argument("--name", type=str, default='SingHop')
    parser.add_argument("--mode", type=str, default='')
    parser.add_argument("--bert_model", type=str, default='../bert/model.en',
                        help='Currently only support bert-base-uncased and bert-large-uncased')
    
    # feature parameters
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--ques_limit', type=int, default=50)
    parser.add_argument('--all_paras', action='store_true')
    
    # learning and log
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_bert_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--early_stop_epoch', type=int, default=0)
    parser.add_argument("--verbose_step", default=50, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
                                                                                                                                                                                                           
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    parser.add_argument("--do_continue", type=int, default=-1, help="Whether to run eval.")
    parser.add_argument('--eval_list', type=int, default=0)
                                                                                                                                                                                                           
    # device
    parser.add_argument("--encoder_gpu", default='1', type=str, help="device to place bert encoder.")
    parser.add_argument("--device", default='0', type=str, help="device to place model.")
    parser.add_argument("--input_dim", type=int, default=768, help="bert-base=768, bert-large=1024")
                                                                                                                                                                                                          
                                                                                                                                                                                                           
    args = parser.parse_args()
    
    MODEL_dict = {"":"bert", "_bl":"bert", "_ab":"albert", "_abl":"albert", "_abxl":"albert",
                  "_rb":"roberta", "_rbl":"roberta", "_rbxl":"roberta"}
    MODEL_FILEs = {"":"../bert/model.en",  "_bl":"../bert/large.en", 
                  "_abl":"../albert/large.en", "_abxl":"../albert/xlarge.en",
                  "_rb":"../roberta/roberta.en", "_rbl":"../roberta/roberta.en"}
    args.model_type = MODEL_dict.get(args.mode,'bert')
    args.bert_model = MODEL_FILEs.get(args.mode,'../bert/model.en')
    
    args.train_file = 'features/train_singhop{}.json'.format(args.mode)
    args.dev_file = 'features/dev_singhop{}.json'.format(args.mode)
    args.train_example_file = 'features/examples_train_singhop{}.json'.format(args.mode)
    args.dev_example_file = 'features/examples_dev_singhop{}.json'.format(args.mode)
    
    args.checkpoint_path = os.path.join(args.name, 'model')                                                                                                                                           
    args.prediction_path = os.path.join(args.name, 'predict')
    
    os.makedirs(args.checkpoint_path, exist_ok=True) 
    os.makedirs(args.prediction_path, exist_ok=True) 
    
    return args

                                                                                                                                                                                                          
if __name__ == "__main__":
    args = set_config()                                                                                                                                                                                    
    writer = open(args.name + '/train.log', 'w', encoding = 'utf-8',buffering = 1)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.bert_model)
    
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
    with open(args.train_file, 'r', encoding = 'utf-8') as fh:
        train_features = json.load(fh)
    with open(args.train_example_file, 'r', encoding = 'utf-8') as fh:
        train_examples = json.load(fh)
    with open(args.dev_file, 'r', encoding = 'utf-8') as fh:
        dev_features = json.load(fh)
    with open(args.dev_example_file, 'r', encoding = 'utf-8') as fh:
        dev_examples = json.load(fh)
        
    Full_Loader = DataIterator(train_features, train_examples, args)
    eval_dataset = DataIterator(dev_features, dev_examples, args, sequential=True)
                                                                                                                                                                                                           
    # Prepare Model
    model = model_class.from_pretrained(args.bert_model)                                                                                                                            
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model = model.module
    model.cuda()

    # Prepare Optimizer
    lr = args.lr                                                                                                                                                                                           
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)                                                                                                                                           
    t_total = args.epochs*int(len(train_features)/args.batch_size/args.gradient_accumulation_steps)
                                                                                                                                                                                                           
    global_step = 1
    if not args.do_eval:                                                                                                                                                                                         
        model.train()
        for epc in trange(start, int(args.epochs), desc="Epoch"):
            tr_loss = 0
            Loader = Full_Loader                                                                                                                                                                           
            Loader.refresh()                                                                                                                                                                               
                                                                                                                                                                                                           
            for step, batch in enumerate(tqdm(Loader, desc="Iteration")):
                batch = {k:v.cuda() if k not in ['ids','q_len','p_idx'] else v for k,v in batch.items()}
                input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch['context_idxs'], batch['context_mask'], batch['segment_idxs'], batch['y1'], batch['y2']
                loss = model(input_ids, token_type_ids, attention_mask, start_positions, end_positions)                                                                                                    
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
                                                                                                                                                                                                           
                if global_step % 1500 == 0 and step>0:
                    add_figure(args.name, writer, global_step, tr_loss/step)                                                                                                                               
                                                                                                                                                                                                           
            # Save a trained model
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc)))
            writer.write('Epoch {} Predict: \n'.format(epc))
            predict(model, eval_dataset, dev_examples, writer,                                                                                                                                 
                os.path.join(args.prediction_path, 'pred_epoch_{}.json'.format(epc)))
    """
    elif args.eval_list <= 0:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "best_model.pth")))
        predict(model, eval_dataset, dev_example_dict, os.path.join(args.prediction_path, 'pred_best.json'))
    else:
        for epc in range(args.eval_list):
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc))))
            predict(model, eval_dataset, dev_examples,writer,                                                                                                                                              
                os.path.join(args.prediction_path, 'pred_epoch_{}.json'.format(epc)))
    """