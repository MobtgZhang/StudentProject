import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
def evaluate_model(model,test_dataloader,device):
    model.eval()
    f1_val = 0.0
    acc_val = 0.0
    pre_val = 0.0
    rec_val = 0.0
    for item in test_dataloader:
        target = item[0].to(device)
        in_ids = item[1].to(device)
        att_masks = item[2].to(device)
        probability = model(in_ids,att_masks)
        predict = torch.argmax(probability,dim=1)
        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        f1_val += f1_score(target,predict)
        acc_val += accuracy_score(target, predict)
        pre_val += precision_score(target, predict)
        rec_val += recall_score(target, predict)
    length = len(test_dataloader)
    f1_val /= length
    acc_val /= length
    pre_val /= length
    rec_val /= length
    return f1_val,acc_val,pre_val,rec_val
def one_hot(vec,n_class):
    new_vec = np.zeros(shape=(vec.shape[0],n_class))
    for k in range(len(vec)):
        new_vec[k,vec[k]] = 1
    return new_vec
def evaluate_multilabel_model(model,test_dataloader,device):
    n_class = model.n_class
    model.eval()
    target_list = []
    predict_list = []
    predicts = []
    for item in test_dataloader:
        id_text = item["text-ids"]
        in_ids = item["token-ids"].to(device)
        att_masks = item["mask-ids"].to(device)
        target = item["label-ids"].to(device)
        probability = model(in_ids, att_masks)
        target = target.detach().cpu().numpy()
        predict = torch.argmax(probability,dim=1).detach().cpu().numpy()
        target_list.append(target)
        predict_list.append(predict)
        predicts += list(zip(id_text,predict.tolist()))
    target_list = np.hstack(target_list)
    predict_list = np.hstack(predict_list)
    tar_vec = one_hot(target_list, n_class)
    pre_vec = one_hot(predict_list, n_class)
    corr = confusion_matrix(target_list,predict_list)
    f1_val = 0.0
    acc_val = 0.0
    pre_val = 0.0
    rec_val = 0.0
    for k in range(n_class):
        f1_val += f1_score(tar_vec[:, k], pre_vec[:, k])
        acc_val += accuracy_score(tar_vec[:, k], pre_vec[:, k])
        pre_val += precision_score(tar_vec[:, k], pre_vec[:, k],zero_division=0)
        rec_val += recall_score(tar_vec[:, k], pre_vec[:, k])

    f1_val = f1_val / n_class
    acc_val = acc_val / n_class
    pre_val = pre_val / n_class
    rec_val = rec_val / n_class

    return (corr,f1_val,acc_val,pre_val,rec_val),predicts
