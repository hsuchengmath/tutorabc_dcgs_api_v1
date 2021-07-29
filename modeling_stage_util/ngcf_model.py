






import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import random
import math



from modeling_stage_util.util_for_ngcf_model import Cold_Start_Prediction

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm




class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.emb_size = 64 #(?)
        self.node_dropout = 0.1
        head_num = 3
        self.mess_dropout = [0.1 for _ in range(head_num)]
        self.norm_adj = norm_adj
        self.layers = [self.emb_size for _ in range(head_num)]
        self.decay = eval('[1e-5]')[0]
        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
        self.predict_layer = nn.Linear((self.emb_size*4), 1).cuda()
        self.ME_layer = nn.Linear(n_item, (self.emb_size*4)).cuda()


    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, items, drop_flag=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]
            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]
            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
        #look up
        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]  
        u_g_embeddings = u_g_embeddings[users, :]
        i_g_embeddings = i_g_embeddings[items, :]
        # interaction
        pair_emb = u_g_embeddings * i_g_embeddings
        prediction = self.predict_layer(pair_emb)
        return torch.sigmoid(prediction.view(-1))

class NGCF_Modeling:
    def __init__(self, n_user, n_item, norm_adj):
        lr = 0.001
        self.epochs = 20
        self.batch_size = 1024
        # model
        self.model = NGCF(n_user, n_item, norm_adj).cuda()
        # build loss func and opt
        self.loss_function = nn.BCEWithLogitsLoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    
    def train(self, train_user, train_item, train_label):
        print('Start to train NCF model!!')
        batch_num = int(len(train_label) / self.batch_size) + 1
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            for i in range(batch_num):
                user = train_user[i*self.batch_size : (i+1)*self.batch_size]
                item = train_item[i*self.batch_size : (i+1)*self.batch_size]
                label = torch.tensor(train_label[i*self.batch_size : (i+1)*self.batch_size]).float().cuda()
                self.model.zero_grad()
                prediction = self.model(user, item)  # shape=(256,) | Real number
                loss = self.loss_function(prediction, label)
                loss.backward()
                self.optimizer.step()
        print('Finish training model!!')  

      
    def recommend(self, user, item, uid2index ,mat2index, rf_model, test_data, mode):
        '''
        status_tag:
        0: old_u, old_m
        1: old_u, new_m
        2: new_u, old_m
        3: new_u, new_m
        '''
        # build cold start object
        csp_obj = Cold_Start_Prediction(test_data, uid2index, mat2index, mode)
        #
        non_cold_start_user, non_cold_start_item, non_cold_start_index = list(), list(), list()
        cold_start_user, cold_start_item, cold_start_index = list(), list(), list()
        status_tag_list = list()
        for i in range(len(user)):
            if user[i] in uid2index and item[i] in mat2index:
                non_cold_start_user.append(uid2index[user[i]])
                non_cold_start_item.append(mat2index[item[i]])
                non_cold_start_index.append(i)
                status_tag_list.append(0)
            elif user[i] in uid2index and item[i] not in mat2index:
                matched_mid = csp_obj.main(new_uid=None, new_mid=item[i], mode='cold_start_item',test_mode = 'normal')
                non_cold_start_user.append(uid2index[user[i]])
                non_cold_start_item.append(mat2index[matched_mid])
                non_cold_start_index.append(i)
                status_tag_list.append(1)
            elif user[i] not in uid2index and item[i] in mat2index:
                matched_uid = csp_obj.main(new_uid=user[i], new_mid=None, mode='cold_start_user',test_mode = 'normal')
                #non_cold_start_user.append(uid2index[matched_uid])
                #non_cold_start_item.append(mat2index[item[i]])
                #non_cold_start_index.append(i)  
                cold_start_user.append(user[i])
                cold_start_item.append(item[i])
                cold_start_index.append(i)
                status_tag_list.append(2)
            else:
                matched_uid, matched_mid = csp_obj.main(new_uid=user[i], new_mid=item[i], mode='cold_start_user_item',test_mode = 'normal')
                #non_cold_start_user.append(uid2index[matched_uid])
                #non_cold_start_item.append(mat2index[matched_mid])
                #non_cold_start_index.append(i)  
                cold_start_user.append(user[i])
                cold_start_item.append(matched_mid)
                cold_start_index.append(i)
                status_tag_list.append(3)
        # non_cold_start part 
        non_cold_start_pred = list(self.model(non_cold_start_user, non_cold_start_item).cpu().detach().numpy())
        # cold_start part
        cold_start_pred = list()
        for i in range(len(cold_start_user)):
            dat = test_data[(test_data['client_sn']==cold_start_user[i]) & (test_data['MaterialID']==cold_start_item[i])]
            dat = np.array(dat)
            if dat.shape[0] != 0:
                predictions = rf_model.predict(dat)
            else:
                predictions = [0]
            predictions = predictions[0]   
            cold_start_pred.append(predictions)
        # intergrate
        predictions_list = list()
        for i in range(len(non_cold_start_pred)):
            predictions_list.append([non_cold_start_pred[i], non_cold_start_index[i]])
        for i in range(len(cold_start_pred)):
            predictions_list.append([cold_start_pred[i], cold_start_index[i]])
        predictions_list = sorted(predictions_list, key=lambda x:x[1])
        predictions_list = [element[0] for element in predictions_list]
        # build R matx
        U2M2P = dict()
        user, item
        for i in range(len(user)):
            u, m = user[i], item[i]
            if u not in U2M2P:
                U2M2P[u] = dict()
            U2M2P[u][m] = predictions_list[i]
        return U2M2P