




class Cold_Start_Prediction:
    def __init__(self, test_data, uid2index_AD, mat2index_AD, mode='Adult'):
        # load organic dat        
        self.user_dat = pd.read_csv('data/user_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.mat_dat = pd.read_csv('data/material_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.mat_dat = self.mat_dat.fillna('None')
        self.mat_dat = self.mat_dat[self.mat_dat['MaterialType']==mode]
        # 
        #self.user_feature = ['Client_Sex','birthday','education','JobClassName','IndustryClassName']
        self.user_feature = ['JobClassName']
        self.data_process_for_user()
        #
        self.user2data = self.build_user2data()
        self.mat2data = self.build_mat2data()
        #
        self.old_uid_set = set(uid2index_AD.keys())
        self.old_mat_set = set(mat2index_AD.keys())
        

    def data_process_for_user(self):
        self.user_dat = self.user_dat.fillna('None')
        self.user_dat['Client_Sex'].replace('N','None')
        self.user_dat['birthday'] = self.user_dat['birthday'].apply(lambda x: transform_date_to_age(x))
        self.user_dat['JobClassName'].replace('Undefined','None')
        self.user_dat['IndustryClassName'].replace('Undefined','None')
        #self.user_dat = self.user_dat[['client_sn']+self.user_feature]

        
    def build_user2data(self):
        user2data = dict()
        feature2data_list = self.user_dat.to_dict(orient='records')
        for element in feature2data_list:
            uid = element['client_sn']
            if uid not in user2data:
                user2data[uid] = set()
            # common feature
            for feature in self.user_feature:
                user2data[uid].add(element[feature])
            # interest tag feature
            #user2data[uid] = user2data[uid] | set(element['user_interest_tag_list'].split('/**/'))
            user2data[uid] = user2data[uid] - {'None'}
        return user2data
            
    def build_mat2data(self):
        mat2data = dict()
        feature2data_list = self.mat_dat.to_dict(orient='records')
        for element in feature2data_list:
            mid = element['MaterialID']
            if mid not in mat2data:
                mat2data[mid]  = set(element['MDCGSID_ENname'].split('/**/'))
        return mat2data


    def matched_old_id_func(self, ent2data, new_id, old_id_set):
        overlap_num2old_id = dict()
        if new_id in ent2data:
            new_id_data = ent2data[new_id]
            for old_id in list(old_id_set):
                if old_id in ent2data:
                    old_id_data = ent2data[old_id]
                    overlap_num = len(set(new_id_data) & set(old_id_data))
                    if overlap_num not in overlap_num2old_id:
                        overlap_num2old_id[overlap_num] = set()
                    overlap_num2old_id[overlap_num].add(old_id)
            max_overlap = max(overlap_num2old_id.keys())
            matched_id = random.sample(list(overlap_num2old_id[max_overlap]), 1)[0]
        else:
            matched_id = random.sample(list(old_id_set), 1)[0]
        return matched_id

        
    def main(self, new_uid=None, new_mid=None, mode=None, test_mode=None):
        if mode == 'cold_start_user':
            matched_uid = \
                self.matched_old_id_func(ent2data=self.user2data, 
                                         new_id=new_uid, 
                                         old_id_set=self.old_uid_set)
            if test_mode != 'normal':
                return random.sample(list(self.old_uid_set), 1)[0]
            else:
                return matched_uid
        elif mode == 'cold_start_item':
            matched_mid = \
                self.matched_old_id_func(ent2data=self.mat2data, 
                                         new_id=new_mid, 
                                         old_id_set=self.old_mat_set)
            if test_mode != 'normal':
                return random.sample(list(self.old_mat_set), 1)[0]
            else:
                return matched_mid
        elif mode == 'cold_start_user_item':
            matched_uid = \
                self.matched_old_id_func(ent2data=self.user2data, 
                                         new_id=new_uid, 
                                         old_id_set=self.old_uid_set)
            matched_mid = \
                self.matched_old_id_func(ent2data=self.mat2data, 
                                         new_id=new_mid, 
                                         old_id_set=self.old_mat_set)
            if test_mode != 'normal':
                matched_uid = random.sample(list(self.old_uid_set), 1)[0]
                matched_mid = random.sample(list(self.old_mat_set), 1)[0]
                return  matched_uid , matched_mid
            else:
                return matched_uid , matched_mid

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
