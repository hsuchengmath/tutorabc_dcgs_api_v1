





import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np

 
class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj,mat_individual_col):
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
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)#.cuda()
        self.inter_layer = nn.Linear(len(mat_individual_col), len(mat_individual_col))#.cuda()
        self.predict_layer = nn.Linear((self.emb_size*4)+len(mat_individual_col), 1)#.cuda()

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

    def forward(self, users, items, inter, drop_flag=True):
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
        inter = self.inter_layer(inter)
        pair_emb = torch.cat((pair_emb, inter), -1)
        prediction = self.predict_layer(pair_emb)
        return torch.sigmoid(prediction.view(-1))

class NGCF_Modeling:
    def __init__(self, n_user, n_item, norm_adj,mat_individual_col, load_model=[False,None]):
        lr = 0.001
        self.epochs = 20
        self.batch_size = 1024
        # model
        if load_model[0] is False:
            self.model = NGCF(n_user, n_item, norm_adj,mat_individual_col)#.cuda()
            # build loss func and opt
            self.loss_function = nn.BCEWithLogitsLoss() 
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = NGCF(n_user, n_item, norm_adj,mat_individual_col)#.cuda()
            self.model.load_state_dict(torch.load(load_model[1]))
            self.model.eval()

    
    def train(self, train_user, train_item, train_label,train_interaction, save_model=False):
        print('Start to train NCF model!!')
        train_interaction = torch.tensor(train_interaction)#.cuda()
        batch_num = int(len(train_label) / self.batch_size) + 1
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            for i in range(batch_num):
                user = train_user[i*self.batch_size : (i+1)*self.batch_size]
                item = train_item[i*self.batch_size : (i+1)*self.batch_size]
                label = torch.tensor(train_label[i*self.batch_size : (i+1)*self.batch_size]).float()#.cuda()
                interaction = train_interaction[i*self.batch_size : (i+1)*self.batch_size]
                self.model.zero_grad()
                prediction = self.model(user, item,interaction)  # shape=(256,) | Real number
                loss = self.loss_function(prediction, label)
                loss.backward()
                self.optimizer.step()
        if save_model[0] is True:
            torch.save(self.model.state_dict(), save_model[1])
        print('Finish training model!!')  


    def recommend(self, test_data, uid2index ,mat2index, rf_model, feature_list):
        test_data = test_data[feature_list + ['client_sn','MaterialID']].dropna()
        old_uid, old_mat = list(uid2index.keys()), list(mat2index.keys())
        new_uid = list(set(test_data['client_sn']) - set(old_uid))
        new_mat = list(set(test_data['MaterialID']) - set(old_mat))
        old_dat = test_data[(test_data['client_sn'].isin(old_uid)) & (test_data['MaterialID'].isin(old_mat))]
        new_dat = test_data[(test_data['client_sn'].isin(new_uid)) | (test_data['MaterialID'].isin(new_mat))]
        new_dat = new_dat.dropna()
        # non-cold_start
        old_dat = old_dat.dropna()
        old_uid = list(old_dat['client_sn'])
        old_mat = list(old_dat['MaterialID'])
        old_uid_model = [uid2index[u] for u in old_uid]
        old_mat_model = [mat2index[m] for m in old_mat]
        interaction = old_dat[feature_list].values.tolist()
        non_cold_start_inter = torch.tensor(interaction)#.cuda()
        non_cold_start_pred = list(self.model(old_uid_model, old_mat_model,non_cold_start_inter).cpu().detach().numpy())
        # cold_start
        new_uid = list(new_dat['client_sn'])
        new_mat = list(new_dat['MaterialID'])    
        predictions_list = list(rf_model.predict(new_dat[feature_list]))
        # build U2M2P UMP_dat
        U2M2P, client_sn_list, MaterialID_list, prob_list, cold_start_or_not = dict(), list(), list(), list(),list()
        for i in range(len(old_uid)):
            if old_uid[i] not in U2M2P:
                U2M2P[old_uid[i]] = dict()
            if old_mat[i] not in U2M2P[old_uid[i]]:
                U2M2P[old_uid[i]][old_mat[i]] = non_cold_start_pred[i]
                client_sn_list.append(old_uid[i])
                MaterialID_list.append(old_mat[i])
                prob_list.append(non_cold_start_pred[i])
        for i in range(len(new_uid)):
            if new_uid[i] not in U2M2P:
                U2M2P[new_uid[i]] = dict()
            if new_mat[i] not in U2M2P[new_uid[i]]:
                U2M2P[new_uid[i]][new_mat[i]] = predictions_list[i]  
                client_sn_list.append(new_uid[i])
                MaterialID_list.append(new_mat[i])
                prob_list.append(predictions_list[i])
        UMP_dat = {'client_sn':client_sn_list,'MaterialID':MaterialID_list,'prob':prob_list}
        UMP_dat = pd.DataFrame(UMP_dat)
        return U2M2P, UMP_dat





import numpy as np
import scipy.sparse as sp




class Adj_Matx_Generator:
    def __init__(self, user_num, item_num, user_item_inter):
        self.user_num = user_num
        self.item_num = item_num
        self.user_id_list = user_item_inter[0]
        self.item_id_list = user_item_inter[1]

    def build_u_v_matrix(self):
        self.R = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        for i in range(len(self.user_id_list)):
            uid, iid = self.user_id_list[i], self.item_id_list[i]
            self.R[uid, iid] = 1
    
    def build_uv_uv_matrix(self):
        self.adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        self.adj_mat = self.adj_mat.tolil()
        R = self.R.tolil()
        self.adj_mat[:self.user_num, self.user_num:] = R
        self.adj_mat[self.user_num:, :self.user_num] = R.T
        self.adj_mat = self.adj_mat.todok()

    def mean_adj_single(self,adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        return norm_adj.tocoo()

    def normalized_adj_single(self, adj):
        # D^-1/2 * A * D^-1/2 (normalized Laplacian)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def check_adj_if_equal(self, adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)
        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        return temp

    def main(self):
        self.build_u_v_matrix()
        self.build_uv_uv_matrix()
        norm_adj_mat = self.mean_adj_single(self.adj_mat + sp.eye(self.adj_mat.shape[0]))
        mean_adj_mat = self.mean_adj_single(self.adj_mat)
        normal_adj_mat = self.normalized_adj_single(self.adj_mat + sp.eye(self.adj_mat.shape[0]))
        return self.adj_mat.tocsr(), norm_adj_mat.tocsr(), normal_adj_mat.tocsr()
