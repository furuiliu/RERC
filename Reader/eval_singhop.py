# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:27:11 2021

@author: 柳
"""

import sys
import json

def F1(a,b,c,d):
    correct = max([0,min(b,d)-max(a,c)])
    if correct == 0:
        return 0.0,0.0,0.0
    pred, recall = correct/(b-a), correct/(d-c)
    return 2*pred*recall/(pred+recall), pred, recall

def Eval(filename):
    F,Pred,Rec = [],[],[]
    with open(filename,'r',encoding='utf-8') as fh:
        Ans = json.load(fh)
    for item in Ans.values():
        f1,pred,rec = F1(int(item['predict'][1]),int(item['predict'][2]),int(item['label'][1]),int(item['label'][2]))
        F.append(f1)
        Pred.append(Pred)
        Rec.append(rec)
    return 'F1: {:.3f}\t Predict: {:.3f}\t Recall: {:.3f}\n'.format(sum(F)/len(F),sum(Pred)/len(F), sum(Rec)/len(F))

Dir = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
for i in range(start,end+1):
    print(i,Eval('pred_epoch_{}.json'.format(i)))