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
    
class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose = False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
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

class MF_ASIPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction1_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction2_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)

                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ASIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS-PS] Reach preset epochs, it seems does not converge.")        

    
    def fit(self, x, y, gamma, tao, G = 4,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer_prediction1 = torch.optim.Adam(
            self.prediction1_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction2 = torch.optim.Adam(
            self.prediction2_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        x_all = generate_total_sample(self.num_users, self.num_items)
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

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.prediction1_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop)

                loss = xent_loss

                optimizer_prediction1.zero_grad()
                loss.backward()
                optimizer_prediction1.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred1] Reach preset epochs, it seems does not converge.")

        early_stop = 0
        last_loss = 1e9
        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.prediction2_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop)

                loss = xent_loss

                optimizer_prediction2.zero_grad()
                loss.backward()
                optimizer_prediction2.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred2] Reach preset epochs, it seems does not converge.")
        
        early_stop = 0
        last_loss = 1e9
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                pred_u1 = self.prediction1_model.forward(x_sampled)
                pred_u2 = self.prediction2_model.forward(x_sampled)

                x_sampled_common = x_sampled[(pred_u1.detach().cpu().numpy() - pred_u2.detach().cpu().numpy()) < tao]
                pred_u3 = self.prediction_model.forward(x_sampled_common)
                sub_y = self.prediction1_model.forward(x_sampled_common)
                
                xent_loss = F.binary_cross_entropy(pred_u3, sub_y.detach())

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()    
    
class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_SNIPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose = False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

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
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()
                sum_inv_prop = torch.sum(inv_prop)

                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")

                xent_loss = xent_loss / sum_inv_prop

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
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
    
    
    
    
class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose = False): 

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        prior_y = y_ips.mean()
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

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()
                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.forward(sub_x, True)  
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[G * idx* batch_size: G * (idx+1)*batch_size]] 

                pred_ul,_,_ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_y = torch.Tensor([prior_y] * G * batch_size).cuda()
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:batch_size], reduction="sum") # e^ui

                ips_loss = (xent_loss - imputation_loss)/batch_size 

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction = "sum") 

                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
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
    
    
class MF_ASIPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction1_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.prediction2_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, tao, batch_size, stop, G = 4,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer_prediction1 = torch.optim.Adam(
            self.prediction1_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction2 = torch.optim.Adam(
            self.prediction2_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        
        one_over_zl_obs = self._compute_IPS(x, y, y_ips)        

        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl_obs[selected_idx].cuda()
                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.prediction1_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach())

                loss = xent_loss

                optimizer_prediction1.zero_grad()
                loss.backward()
                optimizer_prediction1.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 10:
                    print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred1] Reach preset epochs, it seems does not converge.")

        early_stop = 0
        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl_obs[selected_idx].cuda()
                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.prediction2_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach())

                loss = xent_loss

                optimizer_prediction2.zero_grad()
                loss.backward()
                optimizer_prediction2.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 10:
                    print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred2] Reach preset epochs, it seems does not converge.")
        
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                pred_u1 = self.prediction1_model.forward(x_sampled)
                pred_u2 = self.prediction2_model.forward(x_sampled)
                x_sampled_common = x_sampled[(pred_u1.detach().cpu().numpy() - pred_u2.detach().cpu().numpy()) < tao]

                pred_u3 = self.prediction_model.forward(x_sampled_common)
                sub_y = self.prediction1_model.forward(x_sampled_common)
                
                xent_loss = F.binary_cross_entropy(pred_u3, sub_y.detach())

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        # pred = self.sigmoid(pred)
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


class MF_CVIB(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        alpha=0.1, gamma=0.01,
        tol=1e-4, verbose=True):

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred,sub_y)

                # pair wise loss
                x_sampled = x_all[ul_idxs[idx* batch_size:(idx+1)*batch_size]]

                pred_ul,_,_ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                logp_hat = pred.log()

                pred_avg = pred.mean()
                pred_ul_avg = pred_ul.mean()

                info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + self.gamma* torch.mean(pred * logp_hat)

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-CVIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

class MF_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

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

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
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
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
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
    

class MF_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

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

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
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
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2 ) * (1 - 1 / inv_prop.detach())).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
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
        
    
def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True) 



class MF_DIB(nn.Module):
    def __init__(self, num_users, num_items, batch_size,embedding_k=4, *args, **kwargs):
        super(MF_DIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.bias_user_emd = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.unbias_user_emd = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.bias_item_emd = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.unbias_item_emd = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

           
    def fit(self, x, y,
        num_epoch=1000, lr=0.05,  
        alpha=0.1, gamma=0.2,
        lamb=1e-3,tol=1e-5,verbose=False):

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

                sub_user_idx = torch.LongTensor(sub_x[:,0]).cuda()
                sub_item_idx = torch.LongTensor(sub_x[:,1]).cuda()

                user_emb_bias = self.bias_user_emd(sub_user_idx)
                user_emb_unbias = self.unbias_user_emd(sub_user_idx)
                item_emb_bias = self.bias_item_emd(sub_item_idx)
                item_emb_unbias = self.unbias_item_emd(sub_item_idx)
                user_emb =  user_emb_bias + user_emb_unbias
                item_emb =  item_emb_bias + item_emb_unbias

                y_hat_unbias = self.sigmoid(torch.sum(user_emb_unbias.mul(item_emb_unbias), 1))
                y_hat_bias = self.sigmoid(torch.sum(user_emb_bias.mul(item_emb_bias), 1))
                y_hat_all = self.sigmoid(torch.sum(user_emb.mul(item_emb), 1))

                loss_unbias = self.xent_func(y_hat_unbias,sub_y)
                loss_bias = self.xent_func(y_hat_bias,sub_y)
                loss_all = self.xent_func(y_hat_all,sub_y)

                loss = (1 - alpha) * loss_unbias + gamma * loss_bias + alpha * loss_all

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()

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
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        user_emb_unbias = self.unbias_user_emd(user_idx)
        item_emb_unbias = self.unbias_item_emd(item_idx)
        pred = self.sigmoid(torch.sum(user_emb_unbias.mul(item_emb_unbias), 1))
        return pred.detach().cpu().numpy()
    

    
    
class MF_DR_BIAS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

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

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()
                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
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
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
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
    
    
class MF_DR_MSE(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, gamma = 1,
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

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

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
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

                imp_bias_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).sum()
                imp_mrdr_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2 ) * (1 - 1 / inv_prop.detach())).sum()
                imp_loss = gamma * imp_bias_loss + (1-gamma) * imp_mrdr_loss
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
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


class MF_Stable_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, mu = 0, eta = 1, stop = 5,
        num_epoch=1000, batch_size=128, lr=0.05, lr1 = 10, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        mu = torch.Tensor([mu])
        mu.requires_grad_(True)
        
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_propensity = torch.optim.Adam(
            [mu], lr=lr1, weight_decay=lamb)
        
        last_loss = 1e9

        observation = torch.zeros([self.num_users, self.num_items])
        for i in range(len(x)):
            observation[int(x[i][0]),int(x[i][1])] = 1
        observation = observation.reshape(self.num_users * self.num_items)
        
        y1 = []
        for i in range(len(x)):
            if y[i] == 1:
                y1.append(self.num_items * x[i][0] + x[i][1])
        
        
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        
        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y, y1, mu)
        else:
            one_over_zl = self._compute_IPS(x, y, y1, mu, y_ips)
        
        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)].detach()
        
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
            # propensity score
                inv_prop = one_over_zl_obs[selected_idx]                

                sub_y = torch.Tensor(sub_y)

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop.detach()).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()  
                
                
                x_all_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
                x_sampled = x_all[x_all_idx]                  
                    
                imputation_y1 = self.imputation.predict(x_sampled)  
                imputation_y1 = self.sigmoid(imputation_y1)
                
                prop_loss = F.binary_cross_entropy(1/one_over_zl[x_all_idx], observation[x_all_idx], reduction="sum")                
                pred_y1 = self.prediction_model.predict(x_sampled)
                pred_y1 = self.sigmoid(pred_y1)

                imputation_loss = F.binary_cross_entropy(imputation_y1, pred_y1, reduction = "none")
                
                loss = prop_loss + eta * ((1 - observation[x_all_idx] * one_over_zl[x_all_idx]) * (imputation_loss - imputation_loss.mean())).sum() ** 2      
                
                optimizer_propensity.zero_grad()
                loss.backward()
                optimizer_propensity.step()
                
                one_over_zl = self._compute_IPS(x, y, y1, mu, y_ips)        
                one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]
                inv_prop = one_over_zl_obs[selected_idx].detach()                                                
                
                pred = self.prediction_model.forward(sub_x)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight = inv_prop.detach(), reduction="sum")
                xent_loss = (xent_loss)/(inv_prop.detach().sum())
                
                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()      
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Stable-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self, x, y, y1, mu, y_ips=None):
        if y_ips is None:
            y_ips = 1
            print("y_ips is none")
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = (len(x) + mu)/ (x[:,0].max() * x[:,1].max() + 2*mu)
            py1o1 = (y.sum() + mu)/ (len(y) +2*mu)
            py0o1 = 1 - py1o1
            propensity = torch.zeros(self.num_users * self.num_items)
            propensity += (py0o1 * po1) / py0
            propensity[np.array(y1)] = (py1o1 * po1) / py1
            one_over_zl = (1 / propensity)
            
        #one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  


class MF_TDR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.original_model = MF(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x, y, gamma):
        
        x_train = torch.zeros([self.num_users,self.num_items])

        for i in range(len(x)):
            x_train[int(x[i][0]),int(x[i][1])] = 1      
        prediction = (x_train.reshape(self.num_users*self.num_items,1)).type(torch.FloatTensor)

        x_train = self.original_model.complete().type(torch.FloatTensor).detach()
        
        optimizer = torch.optim.SGD([self.linear_1.weight, self.linear_1.bias], lr=1e-3, momentum=0.9)

        last_loss = 1e9
        early_stop = 0
        
        for epoch in range(1000):
            all_idx = np.arange(self.num_users*self.num_items)
            np.random.shuffle(all_idx)
            total_batch = (self.num_users*self.num_items) // self.batch_size
            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x_train = x_train[selected_idx].detach()
                sub_prediction = prediction[selected_idx]
             
                out = self.linear_1(sub_x_train)
                out = self.sigmoid(out)
                loss = self.xent_func(out, sub_prediction)
                
                xent_loss = loss
                
                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()

            if (epoch + 1) % 15 == 0:
                print('*'*10)
                print('epoch {}'.format(epoch+1))
              
            if loss > last_loss:
                early_stop += 1 
            else:
                last_loss = loss
            
            if early_stop >= 5:
                break
        
        x_train = x_train.detach()
        propensity = self.sigmoid(self.linear_1(x_train.detach())) 
        propensity[np.where(propensity.cpu() <= gamma)] = gamma
        one_over_zl = 1 / propensity
        return prediction, one_over_zl  

    def fit(self, x, y, prior_y, gamma, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb) 
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        observation, one_over_zl = self._compute_IPS(x, y, gamma)
        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        prior_y = prior_y.mean()
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                
                # mini-batch training                
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl_obs[selected_idx].detach()

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[G * idx* self.batch_size: G * (idx+1)*self.batch_size]]

                pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")               
                
                imputation_y = torch.Tensor([prior_y]* G *selected_idx.shape[0])
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="sum")

                ips_loss = (xent_loss - imputation_loss)

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y,reduction="sum")

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    
                    print("[MF-TDR] epoch:{}, xent:{}".format(epoch, epoch_loss))

                    e_loss = F.binary_cross_entropy(pred, sub_y, reduction = "none")                    
                    e_hat_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction = "none")
                    
                    TMLE_beta = inv_prop-1
                    TMLE_alpha = e_loss - e_hat_loss
                    TMLE_epsilon = ((TMLE_alpha * TMLE_beta).sum()/(TMLE_beta * TMLE_beta).sum())
                    e_hat_TMLE = TMLE_epsilon.item() * (one_over_zl.float()- torch.tensor([1.])) 
                    e_hat_TMLE_obs = e_hat_TMLE[np.where(observation.cpu() == 1)]
                    
                    np.random.shuffle(all_idx)
                    np.random.shuffle(x_all)
                    
                    selected_idx = all_idx[0:self.batch_size]
                    sub_x = x[selected_idx]
                    sub_y = y[selected_idx]

                    # propensity score
                    inv_prop = one_over_zl_obs[selected_idx].detach()

                    sub_y = torch.Tensor(sub_y)

                    pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  
                    pred = self.sigmoid(pred)

                    x_sampled = x_all[ul_idxs[0: G * self.batch_size]]

                    pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)
                    pred_ul = self.sigmoid(pred_ul)

                    xent_loss = ((F.binary_cross_entropy(pred, sub_y, reduction="none") ** 2) * inv_prop).sum() # o*eui/pui
                        
                    imputation_loss = ((F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="none") + e_hat_TMLE_obs[selected_idx].squeeze().detach()) ** 2).sum()
 
                    ips_loss = (xent_loss - imputation_loss)

                    sub_x_sampled_number = []
                    for i in x_sampled:
                        sub_x_sampled_number.append((self.num_items * i[0] + i[1]))
                    sub_x_sampled_number = np.array(sub_x_sampled_number)
                    
                    direct_loss = ((F.binary_cross_entropy(pred_ul, imputation_y, reduction="none") + e_hat_TMLE[sub_x_sampled_number].squeeze().detach()) ** 2).sum()                    
                    
                    loss = (ips_loss + direct_loss)/sub_x_sampled_number.shape[0]

                    optimizer_prediction.zero_grad()
                    loss.backward()
                    optimizer_prediction.step()
                    
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-TDR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-TDR] Reach preset epochs, it seems does not converge.")
                        
    def predict(self, x):
        pred = self.prediction_model.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()
         

    
class MF_TDR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.original_model = MF(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x, y, gamma):
        
        x_train = torch.zeros([self.num_users,self.num_items])

        for i in range(len(x)):
            x_train[int(x[i][0]),int(x[i][1])] = 1
        prediction = (x_train.reshape(self.num_users*self.num_items,1)).type(torch.FloatTensor)

        x_train = self.original_model.complete().type(torch.FloatTensor).detach()
        
        optimizer = torch.optim.SGD([self.linear_1.weight, self.linear_1.bias], lr=1e-3, momentum=0.9)

        last_loss = 1e9 
        early_stop = 0
        for epoch in range(1000):
            all_idx = np.arange(self.num_users*self.num_items)
            np.random.shuffle(all_idx)
            total_batch = (self.num_users*self.num_items)// self.batch_size

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x_train = x_train[selected_idx].detach()
                sub_prediction = prediction[selected_idx]
             
                out = self.linear_1(sub_x_train)
                out = self.sigmoid(out)
                loss = self.xent_func(out, sub_prediction)
                
                xent_loss = loss
                optimizer.zero_grad()
                xent_loss.backward()

                optimizer.step()

            if (epoch + 1) % 15 == 0:
                print('*'*10)
                print('epoch {}'.format(epoch+1))
            
            if loss > last_loss:
                early_stop += 1 
            else:
                last_loss = loss
            
            if early_stop >= 5:
                break
        
        x_train = x_train.detach()
        propensity = self.sigmoid(self.linear_1(x_train.detach())) 
        propensity[np.where(propensity.cpu() <= gamma)] = gamma
        one_over_zl = 1 / propensity
                      
        return prediction, one_over_zl  
        
    def fit(self, x, y, stop = 1,
        num_epoch=1000,lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
     
        last_loss = 1e9
   
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) 
        total_batch = num_sample // self.batch_size
        observation, one_over_zl = self._compute_IPS(x, y, gamma)

        early_stop = 0
        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        
        for epoch in range(num_epoch):            
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = one_over_zl_obs[selected_idx].detach()                
                
                sub_y = torch.Tensor(sub_y)
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)
                                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation.predict(x_sampled)
                pred_u = self.sigmoid(pred_u)     
                imputation_y1 = self.sigmoid(imputation_y1)          
                                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                        
                ips_loss = xent_loss - imputation_loss 
                                
                # direct loss                                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")             

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.imputation.forward(sub_x)
                imputation_y = self.sigmoid(imputation_y)
                    
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-TDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    
                    e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                    e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")   
                    
                    TMLE_beta = inv_prop-1
                    TMLE_alpha = e_loss - e_hat_loss
                    TMLE_epsilon = ((TMLE_alpha * TMLE_beta).sum()/(TMLE_beta * TMLE_beta).sum())
                    e_hat_TMLE = TMLE_epsilon.item() * (one_over_zl.float()- torch.tensor([1.]))
                    e_hat_TMLE_obs = e_hat_TMLE[np.where(observation.cpu() == 1)]

                    np.random.shuffle(x_all)
                    np.random.shuffle(all_idx)
                    
                    selected_idx = all_idx[0:self.batch_size]
                    sub_x = x[selected_idx] 
                    sub_y = y[selected_idx]

                    inv_prop = one_over_zl_obs[selected_idx].detach()                
                
                    sub_y = torch.Tensor(sub_y)
                       
                    pred = self.prediction_model.forward(sub_x)
                    imputation_y = self.imputation.predict(sub_x)
                    pred = self.sigmoid(pred)
                    imputation_y = self.sigmoid(imputation_y)
                               
                    x_sampled = x_all[ul_idxs[0 : G*self.batch_size]]
                                       
                    pred_u = self.prediction_model.forward(x_sampled) 
                    imputation_y1 = self.imputation.predict(x_sampled)
                    pred_u = self.sigmoid(pred_u)     
                    imputation_y1 = self.sigmoid(imputation_y1)                             
                
                    xent_loss = ((F.binary_cross_entropy(pred, sub_y, reduction ="none") ** 2) * inv_prop).sum()
                    imputation_loss = ((F.binary_cross_entropy(pred, imputation_y, reduction="none")
                                        + e_hat_TMLE_obs[selected_idx].squeeze().detach()) ** 2).sum()
                        
                    ips_loss = xent_loss - imputation_loss
                    
                    # direct loss
                    sub_x_sampled_number = []
                    for i in x_sampled:
                        sub_x_sampled_number.append((self.num_items * i[0] + i[1]))
                    sub_x_sampled_number = np.array(sub_x_sampled_number)                 
                
                    direct_loss = ((F.binary_cross_entropy(pred_u, imputation_y1, reduction="none") + e_hat_TMLE[sub_x_sampled_number].squeeze().detach()) ** 2).sum()
                    
                    loss = (ips_loss + direct_loss)/sub_x_sampled_number.shape[0]
                    
                    optimizer_prediction.zero_grad()
                    loss.backward()
                    optimizer_prediction.step()
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-TDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-TDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy() 
    
    


#########################




class MF_IPS_V2(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)       
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1, eta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)                
                sub_y = torch.Tensor(sub_y).cuda()                      
                pred = self.prediction_model.forward(sub_x)           
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       

                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)            
                ips_loss = xent_loss 
                             
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()
                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    

                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                ones_all = torch.ones(len(inv_prop_all)).cuda()
                w_all = torch.divide(sub_obs,1/inv_prop_all)-torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
                bmse_loss = (torch.mean(w_all * pred))**2
                
                loss = alpha * prop_loss + beta * pred_loss + ips_loss +  eta * bmse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()



class MF_DR_V2(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)        
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1, eta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).cuda()                
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled).cuda()             
                
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                imputation_loss = -torch.sum(imputation_y * torch.log(pred + 1e-6) + (1-imputation_y) * torch.log(1 - pred + 1e-6))
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                # direct loss
                                
                direct_loss = -torch.sum(imputation_y1 * torch.log(pred_u + 1e-6) + (1-imputation_y1) * torch.log(1 - pred_u + 1e-6))
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                                                  
                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = -sub_y * torch.log(pred + 1e-6) - (1-sub_y) * torch.log(1 - pred + 1e-6)
                e_hat_loss = -imputation_y * torch.log(pred + 1e-6) - (1-imputation_y) * torch.log(1 - pred + 1e-6)
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()

                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                ones_all = torch.ones(len(inv_prop_all)).cuda()
                w_all = torch.divide(sub_obs,1/inv_prop_all)-torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
                bmse_loss = (torch.mean(w_all * pred))**2
                
                loss = alpha * prop_loss + beta * pred_loss + theta * imp_loss + dr_loss + eta * bmse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
        
        torch.save(self.propensity_model.state_dict(), 'weight_model0.pth')
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    




class MF_MBIPS(nn.Module):
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


    def fit(self, x, y, y_ips=None, G=5, gamma = 1,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05,lamb2=0,J=3,
        tol=1e-4, verbose = False):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_weight = torch.optim.Adam(
            self.weight_model.parameters(), lr=lr2, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        # for kuai 
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

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                w_tran = (w.unsqueeze(1)).T
      
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
            
                loss_ker_bal = 0
                for i in range(J):
                    obs_moment_i = (feature_emd_o) ** (i+1)
                    all_moment_i = (feature_emd_d) ** (i+1)                                
                    moment_loss = self.mse( 1/len(w) * torch.mm(w_tran,obs_moment_i).squeeze(),torch.mean(all_moment_i,0))
                    loss_ker_bal = loss_ker_bal + moment_loss          

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
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
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
    
    
    
class MF_MBDR(nn.Module):
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

    def fit(self, x, y,y_ips=None,gamma = 1,J=3,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, lr2=0.05, lamb2=0, 
        tol=1e-4, G=5, verbose = False): 

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

                # weight model 
                w = self.weight_model.predict(sub_x).cuda()
                w = w * inv_prop
                w_tran = (w.unsqueeze(1)).T
      
                feature_emd_o = self.get_embedding(sub_x)
                feature_emd_d = self.get_embedding(x_sampled)
            
                loss_ker_bal = 0
                for i in range(J):
                    obs_moment_i = (feature_emd_o) ** (i+1)
                    all_moment_i = (feature_emd_d) ** (i+1)                                
                    moment_loss = self.mse(1/len(w) * torch.mm(w_tran,obs_moment_i).squeeze(),torch.mean(all_moment_i,0))
                    loss_ker_bal = loss_ker_bal + moment_loss                      

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
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")
                

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
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
    
