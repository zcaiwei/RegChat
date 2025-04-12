import numpy as np
import scipy.sparse as sp
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
import torch
import torch.nn as nn
import glob
from DGI_HGCN.models import DGI, LogReg
from DGI_HGCN.utils import process
import pickle as pkl
import ast
import pandas as pd
import random
import sys



def preprocess_features_dense(features):
    """Row-normalize feature matrix and return representation for dense input"""
    rowsum = np.array(features.sum(1))
    epsilon = 1e-7
    r_inv = np.power(rowsum + epsilon, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    dense_features = r_mat_inv.dot(features)
    return dense_features


def train(features, labels, patience, nb_class, hid_units, shid,  
          nonlinearity, lr, l2_coef, adjs, sparse, nb_epochs, batch_size, 
          hgnn_epoch, metapath_weight, fixed_idx, original_idx_len, train_idx_len,
          test_idx_len, cuda_num, fold_epochs, save_dir):
    '''HDGI model train'''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # param def1
    if torch.cuda.is_available():
        device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
        features = torch.tensor(features).to(device)
        metapath_weight = torch.tensor(metapath_weight).to(device)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = nb_class
    P = int(len(adjs))

    # labels one hot encoding
    sample_categories = labels.numpy()
    one_hot_labels = np.zeros((nb_nodes, nb_classes))
    for i in range(nb_nodes):
        one_hot_labels[i, sample_categories[i]] = 1
    labels = one_hot_labels

    '''data processing'''
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = one_hot_labels.shape[1]
    P = int(len(adjs))
    
    # data idx sep----->train, test, val
    original = range(original_idx_len)
    idx_train = random.sample(original, train_idx_len)
    original = list(set(original) ^ set(idx_train))
    idx_val = random.sample(original, test_idx_len)
    idx_test = list(set(original) ^ set(idx_val))
    
    # adjs normalzation
    nor_adjs = []
    sp_nor_adjs = []
    for adj in adjs:
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        
        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_nor_adjs.append(sp_adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
            adj = adj[np.newaxis]
            nor_adjs.append(adj)
    features = features.cpu().detach().numpy()
    features = torch.FloatTensor(features[np.newaxis])
    if sparse:
        sp_nor_adjs = torch.stack(sp_nor_adjs)
    else:
        nor_adjs = torch.FloatTensor(np.array(nor_adjs)).float()
    size = sys.getsizeof(adjs)     
    print(f'The variable adjs occupies {size} bytes in memory.')    
    del adjs
    size = sys.getsizeof(sp_nor_adjs)
    print(f'The variable sp_nor_adjs occupies {size} bytes in memory.')
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    '''load model for pretraining'''
    before_memory = torch.cuda.memory_allocated()
    model = DGI(ft_size, hid_units, shid, P, nonlinearity, metapath_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    after_memory = torch.cuda.memory_allocated()
    model_memory = after_memory - before_memory 
    print(f'Model memory usage: {model_memory} bytes')

    if torch.cuda.is_available():
        device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
        print('Using CUDA')
        model.to(device)
        features = features.to(device)
        if sparse:
            sp_nor_adjs = sp_nor_adjs.to(device)
        else:
            nor_adjs = nor_adjs.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
    
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    attention1_list = []
    attention2_list = []
    
    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()
        idx = list(range(nb_nodes))
        non_fixed_positions = [i for i in range(nb_nodes) if i not in fixed_idx]
        non_fixed_idx = np.random.permutation(non_fixed_positions)
        j = 0
        for i in range(nb_nodes):
            if i not in fixed_idx:
                idx[i] = non_fixed_idx[j]
                j += 1
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.to(device)
            lbl = lbl.to(device)
        
        attention1, attention2, logits = model(features, shuf_fts, sp_nor_adjs if sparse else nor_adjs, sparse, None, None, None)
        attention1_list.append(attention1.cpu().detach().numpy())

        loss = b_xent(logits, lbl)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_dgi.pkl'))
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()
        torch.cuda.empty_cache()
        
        model.eval()
        with torch.no_grad():
            embeds, _ = model.embed(features, sp_nor_adjs if sparse else nor_adjs, sparse, None)
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            
            log = LogReg(hid_units, nb_classes)
            if torch.cuda.is_available():
                log.to(device)
            log.train()
            
            train_lbls = torch.argmax(labels[0, idx_train], dim=1)
            val_lbls = torch.argmax(labels[0, idx_val], dim=1)
            
            train_logits = log(train_embs)
            val_logits = log(val_embs)
            train_preds = torch.argmax(train_logits, dim=1)
            val_preds = torch.argmax(val_logits, dim=1)
            train_acc = torch.sum(train_preds == train_lbls).float() / train_lbls.shape[0]
            val_acc = torch.sum(val_preds == val_lbls).float() / val_lbls.shape[0]
        print(f"Epoch: [{epoch+1}/{nb_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc.item():.4f}, Val Accuracy: {val_acc.item():.4f}")

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_dgi.pkl')))

    embeds, _ = model.embed(features, sp_nor_adjs if sparse else nor_adjs, sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1).to(device)
    tot_mac = 0
    accs = []
    mac_f1 = []

    for _ in range(5):
        bad_counter = 0
        best = 10000
        loss_values = []
        best_epoch = 0
        patience = 20
        
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if torch.cuda.is_available():
            log.to(device)
        loss_list = []
        epochs = fold_epochs 
        for epoch in range(epochs):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss_list.append(loss.item())
            logits_val = log(val_embs)
            loss_val = xent(logits_val, val_lbls)
            loss_values.append(loss_val)
            loss.backward()
            opt.step()
            torch.save(log.state_dict(), os.path.join(save_dir, '{}.mlp.pkl'.format(epoch)))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
            
            if bad_counter == patience:
                break
            
            files = glob.glob(os.path.join(save_dir, '*.mlp.pkl'))
            for file in files:
                epoch_nb = int(file.split(os.sep)[-1].split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)
            
        files = glob.glob(os.path.join(save_dir, '*.mlp.pkl'))
        for file in files:
            epoch_nb = int(file.split(os.sep)[-1].split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)
        
        print("Optimization Finished!")  
        print('Loading {}th epoch'.format(best_epoch))
        log.load_state_dict(torch.load(os.path.join(save_dir, '{}.mlp.pkl'.format(best_epoch))))
        
        files = glob.glob(os.path.join(save_dir, '*.mlp.pkl'))
        for file in files:
                os.remove(file)
        
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc)
        mac = torch.Tensor(np.array(process.macro_f1(preds, test_lbls))) 
        mac_f1.append(mac)
        
    accs = torch.stack(accs)
    print('Average accuracy:', accs.mean())
    print('accuracy std:', accs.std())
    mac_f1 = torch.stack(mac_f1)
    print('Average mac_f1:', mac_f1.mean())
    print('mac_f1 std:', mac_f1.std())
    
    return attention1_list, attention2_list, min(loss_list), embeds


