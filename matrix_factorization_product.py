# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time

from utils import ndcg_func,  recall_func, precision_func
acc_func = lambda x,y: np.sum(x == y) / len(x)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class MF(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        nn.init.uniform_(self.W.weight, a = 0, b = 1)
        nn.init.uniform_(self.H.weight, a = 0, b = 1)


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.sigmoid(self.linear_1(z_emb))

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size / 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size / 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
    
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        
        return torch.squeeze(x)    
    


###### Kernel Balancing with Gaussian kernel ######


class MF_KBIPS_Gau(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def gau_kernel(self,X,Y,gamma = 0.1):
        if X.ndim == 1: #dim of X, if dim_x = 1
            X_norm_squared = X **2
            Y_norm_squared = Y **2
        else: #dim >= 2
            X_norm_squared = (X **2).sum(axis = 1).reshape(X.shape[0],1)
            Y_norm_squared = (Y **2).sum(axis = 1).reshape(Y.shape[0],1)
        squared_Euclidean_distances = Y_norm_squared[:,] + X_norm_squared.T - 2 * torch.mm(Y,X.T)
        return torch.exp(-squared_Euclidean_distances * gamma)


    def fit(self, x, y, y_ips=None, G=4, gamma = 1, C = 1e-5,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,J=3,
        tol=1e-4, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                
                inv_prop = one_over_zl[selected_idx].cuda()           
        
                rand_num = np.random.randint(low=0,high=batch_size * G,size=J) # random sample

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[rand_num])       
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                
                h_all = torch.mean(self.gau_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.gau_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
                
                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal
                
                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # prediction model

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,weight=w)

                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-KBIPS-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-KBIPS-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-KBIPS-Gau] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()
    
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl
    
    

class MF_AKBIPS_Gau(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def gau_kernel(self,X,Y,gamma = 0.1):
        if X.ndim == 1: #dim of X, if dim_x = 1
            X_norm_squared = X **2
            Y_norm_squared = Y **2
        else: #dim >= 2
            X_norm_squared = (X **2).sum(axis = 1).reshape(X.shape[0],1)
            Y_norm_squared = (Y **2).sum(axis = 1).reshape(Y.shape[0],1)
        squared_Euclidean_distances = Y_norm_squared[:,] + X_norm_squared.T - 2 * torch.mm(Y,X.T)
        return torch.exp(-squared_Euclidean_distances * gamma)


    def fit(self, x, y, y_ips=None, G=4, gamma = 1, C = 1e-5,num_w_epo = 3,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,
        lr3 = 0.05,lamb3=0,J=3,tol=1e-5, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb3)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]

                inv_prop = one_over_zl[selected_idx].cuda() 

                # fit e with k(,)
                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)
                e_loss = F.binary_cross_entropy(pred, sub_y,reduction = 'none')

                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)

                loss_epsilon = (((e_loss - torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum()
                if (loss_epsilon_last - loss_epsilon)/(loss_epsilon_last + 1e-10) < 0.2:
                    for i in range(num_w_epo):
                        loss_epsilon = (((e_loss - torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum()
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()
                loss_epsilon_last =  loss_epsilon 
                
                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()


                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])                       

                h_all = torch.mean(self.gau_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.gau_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))

                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # prediction model

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                # sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)

                xent_loss = F.binary_cross_entropy(pred, sub_y,weight=w)
                
                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-AKBIPS-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-AKBIPS-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-AKBIPS-Gau] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl     
    
    
    
    
class MF_WKBIPS_Gau(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))  
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def gau_kernel(self,X,Y,gamma = 0.1):
        if X.ndim == 1: #dim of X, if dim_x = 1
            X_norm_squared = X **2
            Y_norm_squared = Y **2
        else: #dim >= 2
            X_norm_squared = (X **2).sum(axis = 1).reshape(X.shape[0],1)
            Y_norm_squared = (Y **2).sum(axis = 1).reshape(Y.shape[0],1)
        squared_Euclidean_distances = Y_norm_squared[:,] + X_norm_squared.T - 2 * torch.mm(Y,X.T)
        return torch.exp(-squared_Euclidean_distances * gamma)


    def fit(self, x, y, y_ips=None, G=4, gamma = 1, C = 1e-5,num_w_epo = 3,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,
        lr3=0.05,lamb3=0,J=3,tol=1e-4, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]

                inv_prop = one_over_zl[selected_idx].cuda() 

                #  k(,)
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                w_all = torch.ones(G*batch_size).cuda()


                worst_loss = 1/len(x_sampled) *  (torch.dot(w , torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                torch.dot(w_all , torch.mm(self.epsilon, self.gau_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d))).sum()
                loss_epsilon = - worst_loss/norm
                if (loss_epsilon - loss_epsilon_last)/(loss_epsilon_last + 1e-10) < 0.5:
                    for i in range(num_w_epo):
                        worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                        torch.dot(w_all , torch.mm(self.epsilon, self.gau_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                        norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d))).sum()
                        loss_epsilon = - worst_loss/norm
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()       
                loss_epsilon_last =  loss_epsilon

    
                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()
                

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])      
                

                h_all = torch.mean(self.gau_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.gau_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))

                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal
                

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # prediction model

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)
                xent_loss = F.binary_cross_entropy(pred, sub_y,weight=w)


                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-WKBIPS-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-WKBIPS-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-WKBIPS-Gau] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  



##########
class MF_KBDR_Gau(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def gau_kernel(self,X,Y,gamma = 0.1):
        if X.ndim == 1: #dim of X, if dim_x = 1
            X_norm_squared = X **2
            Y_norm_squared = Y **2
        else: #dim >= 2
            X_norm_squared = (X **2).sum(axis = 1).reshape(X.shape[0],1)
            Y_norm_squared = (Y **2).sum(axis = 1).reshape(Y.shape[0],1)
        squared_Euclidean_distances = Y_norm_squared[:,] + X_norm_squared.T - 2 * torch.mm(Y,X.T)
        return torch.exp(-squared_Euclidean_distances * gamma)

    def fit(self, x, y,y_ips=None,gamma = 1,C=1e-2,J=3,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, lr2=0.05, lamb2=0, 
        tol=1e-4, G=4, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                
                inv_prop = one_over_zl[selected_idx].cuda()

                rand_num = np.random.randint(low=0,high=batch_size * G,size=J) # random sample

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                feature_emd_j = self.get_embedding(x_sampled[rand_num])       
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                        
                h_all = torch.mean(self.gau_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.gau_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
                
                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # propensity score
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                sub_y = torch.Tensor(sub_y).cuda()                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()                                                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()               
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=w, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                   
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * w).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-KBDR-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-KBDR-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-KBDR-Gau] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl 




class MF_AKBDR_Gau(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature


    def gau_kernel(self,X,Y,gamma = 0.1):
        if X.ndim == 1: #dim of X, if dim_x = 1
            X_norm_squared = X **2
            Y_norm_squared = Y **2
        else: #dim >= 2
            X_norm_squared = (X **2).sum(axis = 1).reshape(X.shape[0],1)
            Y_norm_squared = (Y **2).sum(axis = 1).reshape(Y.shape[0],1)
        squared_Euclidean_distances = Y_norm_squared[:,] + X_norm_squared.T - 2 * torch.mm(Y,X.T)
        return torch.exp(-squared_Euclidean_distances * gamma)

    def fit(self, x, y, y_ips,gamma = 1,C=1e-2,num_w_epo = 3,J=3,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0,lr1=0.05, lamb1=0, lr2=0.05, lamb2=0, lr3=0.05, lamb3=0, 
        tol=1e-4, G=4, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb3)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0      
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                inv_prop = one_over_zl[selected_idx].cuda() 

                # fit e with k(,)
                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x) 
                imputation_y = self.imputation.forward(sub_x)
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)

                loss_epsilon = (((e_loss- e_hat_loss - torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum() 
                if (loss_epsilon_last - loss_epsilon)/(loss_epsilon_last + 1e-10) < 0.4:
                    for i in range(num_w_epo):
                        loss_epsilon = (((e_loss- e_hat_loss - torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum() 
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()
                loss_epsilon_last =  loss_epsilon

                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])           
                h_all = torch.mean(self.gau_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.gau_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
                
                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal
                
                optimizer_weight.zero_grad()
                loss_weight_model.backward(retain_graph=True)
                optimizer_weight.step()
                

                # propensity score
                w = self.weight_model.predict(sub_x).cuda()  
                w = w * inv_prop
                
                # sub_y = torch.Tensor(sub_y).cuda()                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()                                                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=w, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                   
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * w).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-AKBDR-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-AKBDR-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-AKBDR-Gau] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  



class MF_WKBDR_Gau(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))     
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def gau_kernel(self,X,Y,gamma = 0.1):
        if X.ndim == 1: #dim of X, if dim_x = 1
            X_norm_squared = X **2
            Y_norm_squared = Y **2
        else: #dim >= 2
            X_norm_squared = (X **2).sum(axis = 1).reshape(X.shape[0],1)
            Y_norm_squared = (Y **2).sum(axis = 1).reshape(Y.shape[0],1)
        squared_Euclidean_distances = Y_norm_squared[:,] + X_norm_squared.T - 2 * torch.mm(Y,X.T)
        return torch.exp(-squared_Euclidean_distances * gamma)

    def fit(self, x, y,y_ips,gamma = 1,C=1e-2,num_w_epo = 3,J=3,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, lr2=0.05, lamb2=0, lr3=0.05, lamb3=0, 
        tol=1e-4, G=4, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb3)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                
                inv_prop = one_over_zl[selected_idx].cuda() 

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                w_all = torch.ones(G*batch_size).cuda()

                worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                torch.dot(w_all , torch.mm(self.epsilon, self.gau_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d))).sum()
                loss_epsilon = - worst_loss/norm
                
                if (loss_epsilon - loss_epsilon_last)/(loss_epsilon_last + 1e-10) < 0.5:
                    for i in range(num_w_epo):
                        worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                        torch.dot(w_all , torch.mm(self.epsilon, self.gau_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                        norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.gau_kernel(feature_emd_o,feature_emd_d))).sum()
                        loss_epsilon = - worst_loss/norm
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()       
                loss_epsilon_last =  loss_epsilon

                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()
                

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])           
                h_all = torch.mean(self.gau_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.gau_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
                
                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # propensity score
                w = self.weight_model.predict(sub_x).cuda() 
                w = w * inv_prop
                sub_y = torch.Tensor(sub_y).cuda()                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()                                                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()               
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=w, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                   
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * w).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-WKBDR-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-WKBDR-Gau] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-WKBDR-Gau] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl






##### Kernel Balancing with Exponential kernel #####
    




class MF_KBIPS_Exp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature


    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)



    def fit(self, x, y, y_ips=None, G=4, gamma = 1, C = 1e-5,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,J=3,
        tol=1e-4, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                
                inv_prop = one_over_zl[selected_idx].cuda()           
                rand_num = np.random.randint(low=0,high=batch_size * G,size=J) # random sample

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[rand_num])       
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                

                h_all = torch.mean(self.exp_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.exp_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))

                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # prediction model

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)

                xent_loss = F.binary_cross_entropy(pred, sub_y,weight=w)

                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-KBIPS-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-KBIPS-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-KBIPS-Exp] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()
    
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl
    
    

class MF_AKBIPS_Exp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature


    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)


    def fit(self, x, y, y_ips=None, G=4, gamma = 1, C = 1e-5,num_w_epo = 3,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,
        lr3 = 0.05,lamb3=0,J=3,tol=1e-5, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb3)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]

                inv_prop = one_over_zl[selected_idx].cuda() 

                # fit e with k(,)
                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)
                e_loss = F.binary_cross_entropy(pred, sub_y,reduction = 'none')

                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)

                loss_epsilon = (((e_loss - torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum()
                if (loss_epsilon_last - loss_epsilon)/(loss_epsilon_last + 1e-10) < 0.2:
                    for i in range(num_w_epo):
                        loss_epsilon = (((e_loss - torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum()
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()
                loss_epsilon_last =  loss_epsilon 
                
                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop


                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])                       

                h_all = torch.mean(self.exp_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.exp_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))

                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # prediction model

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                # sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)

                xent_loss = F.binary_cross_entropy(pred, sub_y,weight=w)
                
                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-AKBIPS-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-AKBIPS-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-AKBIPS-Exp] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl     
    
    
    
    
class MF_WKBIPS_Exp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))  
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature


    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)



    def fit(self, x, y, y_ips=None, G=4, gamma = 1, C = 1e-5,num_w_epo = 3,
        num_epoch=100, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,
        lr3=0.05,lamb3=0,J=3,tol=1e-4, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]

                inv_prop = one_over_zl[selected_idx].cuda() 

                #  k(,)
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                w_all = torch.ones(G*batch_size).cuda()

                worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                torch.dot(w_all , torch.mm(self.epsilon, self.exp_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d))).sum()
                loss_epsilon = - worst_loss/norm
                if (loss_epsilon - loss_epsilon_last)/(loss_epsilon_last + 1e-10) < 0.5:
                    for i in range(num_w_epo):
                        worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                        torch.dot(w_all , torch.mm(self.epsilon, self.exp_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                        norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d))).sum()
                        loss_epsilon = - worst_loss/norm
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()       
                loss_epsilon_last =  loss_epsilon

            
                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()    

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])      
                

                h_all = torch.mean(self.exp_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.exp_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))

                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal
                
                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # prediction model

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x)

                xent_loss = F.binary_cross_entropy(pred, sub_y,weight=w)

                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-WKBIPS-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-WKBIPS-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-WKBIPS-Exp] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  



##########
class MF_KBDR_Exp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)


    def fit(self, x, y,y_ips=None,gamma = 1,C=1e-2,J=3,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, lr2=0.05, lamb2=0, 
        tol=1e-4, G=4, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                
                inv_prop = one_over_zl[selected_idx].cuda()
                rand_num = np.random.randint(low=0,high=batch_size * G,size=J) # random sample

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                feature_emd_j = self.get_embedding(x_sampled[rand_num])       
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                        
                h_all = torch.mean(self.exp_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.exp_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
                
                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # propensity score
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                sub_y = torch.Tensor(sub_y).cuda()                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()                                                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()               
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=w, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                   
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * w).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-KBDR-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-KBDR-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-KBDR-Exp] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl 




class MF_AKBDR_Exp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature


    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)


    def fit(self, x, y, y_ips,gamma = 1,C=1e-2,num_w_epo = 3,J=3,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0,lr1=0.05, lamb1=0, lr2=0.05, lamb2=0, lr3=0.05, lamb3=0, 
        tol=1e-4, G=4, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb3)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0      
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]

                inv_prop = one_over_zl[selected_idx].cuda() 

                # fit e with k(,)
                sub_y = torch.Tensor(sub_y).cuda()
                pred = self.prediction_model.forward(sub_x) 
                imputation_y = self.imputation.forward(sub_x)
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)


                loss_epsilon = (((e_loss- e_hat_loss - torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum() 
                if (loss_epsilon_last - loss_epsilon)/(loss_epsilon_last + 1e-10) < 0.4:
                    for i in range(num_w_epo):
                        loss_epsilon = (((e_loss- e_hat_loss - torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum() 
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()
                loss_epsilon_last =  loss_epsilon    


                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)               
                topj_indices = topj_indices.cpu().numpy().tolist()

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])           
                h_all = torch.mean(self.exp_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.exp_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))

                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal
                
                optimizer_weight.zero_grad()
                loss_weight_model.backward(retain_graph=True)
                optimizer_weight.step()
    

                # propensity score
                w = self.weight_model.predict(sub_x).cuda()  
                w = w * inv_prop
                
                # sub_y = torch.Tensor(sub_y).cuda()                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()                                                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=w, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                   
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * w).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-AKBDR-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-AKBDR-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-AKBDR-Exp] Reach preset epochs, it seems does not converge.")
        

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  



class MF_WKBDR_Exp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.weight_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = torch.nn.Parameter(torch.rand(1,8192))     
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def get_embedding(self,x):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]

        return feature

    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)


    def fit(self, x, y,y_ips,gamma = 1,C=1e-2,num_w_epo = 3,J=3,
        num_epoch=100, batch_size=128, lr=0.05, lamb=0, lr2=0.05, lamb2=0, lr3=0.05, lamb3=0, 
        tol=1e-4, G=4, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr3, weight_decay=lamb3)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        self.W.load_state_dict(torch.load('data/kuai/kuai_user.pth'))
        self.H.load_state_dict(torch.load('data/kuai/kuai_item.pth'))

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        loss_epsilon_last = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                
                inv_prop = one_over_zl[selected_idx].cuda() 

                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
                w_all = torch.ones(G*batch_size).cuda()


                worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                torch.dot(w_all , torch.mm(self.epsilon, self.exp_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d))).sum()
                loss_epsilon = - worst_loss/norm
                
                if (loss_epsilon - loss_epsilon_last)/(loss_epsilon_last + 1e-10) < 0.5:
                    for i in range(num_w_epo):
                        worst_loss = 1/len(x_sampled) * (torch.dot(w , torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) - 
                                        torch.dot(w_all , torch.mm(self.epsilon, self.exp_kernel(feature_emd_d,feature_emd_d)).squeeze())) ** 2
                        norm = 1/len(x_sampled) * torch.mm(torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d)).T, torch.mm(self.epsilon, self.exp_kernel(feature_emd_o,feature_emd_d))).sum()
                        loss_epsilon = - worst_loss/norm
                        optimizer_epo.zero_grad()
                        loss_epsilon.backward(retain_graph=True)
                        optimizer_epo.step()       
                loss_epsilon_last =  loss_epsilon 

                
                topj_values, topj_indices = torch.topk(torch.abs(self.epsilon), k=J)
                topj_indices = topj_indices.cpu().numpy().tolist()
                

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop

                feature_emd_j = self.get_embedding(x_sampled[topj_indices[0]])           
                h_all = torch.mean(self.exp_kernel(feature_emd_j,feature_emd_d),axis =0)
                w_tran = (w.unsqueeze(1)).T
                h_obs = 1/len(w) * (torch.mm(w_tran,self.exp_kernel(feature_emd_j,feature_emd_o))).squeeze()
                zero = torch.zeros(1).cuda()
                loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
                
                loss_weight_model =  1/len(w) * torch.dot(w,torch.log(w)) + gamma * loss_ker_bal

                optimizer_weight.zero_grad()
                loss_weight_model.backward()
                optimizer_weight.step()

                # propensity score
                w = self.weight_model.predict(sub_x).cuda() 
                w = w * inv_prop
                sub_y = torch.Tensor(sub_y).cuda()                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()                                                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()               
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=w, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                   
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * w).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-WKBDR-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-WKBDR-Exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-WKBDR-Exp] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)

        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl
