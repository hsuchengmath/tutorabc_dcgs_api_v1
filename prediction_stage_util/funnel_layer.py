
import random
import numpy as np


 
class Greedy_Based_Funnel_Layer:

    def __init__(self, user_list, mat_list, con_list):
        self.user_list = user_list
        self.mat_list = mat_list
        self.con_list = con_list
        self.group_len_to_user = 24
        self.group_len_to_mat = 25


    def calculate_similarity(self, super_emb1, super_emb2, other_emb1, other_emb2):
        super_emb1 = np.array(super_emb1)
        other_emb1 = np.array(other_emb1)
        super_emb2 = np.array(super_emb2)
        other_emb2 = np.array(other_emb2)   
        mse1 = ((super_emb1 - other_emb1)**2).mean(axis=0)     
        mse2 = ((super_emb2 - other_emb2)**2).mean(axis=0)    
        similarity = mse1 + mse2
        return similarity


    def build_batch_size_list(self):
        remain_num = len(self.user_list) % self.group_len_to_user
        batch_num = int(len(self.user_list) / self.group_len_to_user)
        if remain_num == 0:
            batch_size_list = [self.group_len_to_user for _ in range(batch_num)]
        else:
            batch_size_list = list()
            for i in range(batch_num):
                if i < remain_num:
                    batch_size_list.append(self.group_len_to_user + 1)
                else:
                    batch_size_list.append(self.group_len_to_user)
        return batch_size_list

    def _to_user(self, U2M2P, U2C2P):
        # build batch_size_list
        batch_size_list = self.build_batch_size_list()
        # build sub_matx obj list
        error = False
        sub_U2M2P_list, sub_U2C2P_list, sub_M2U2P_list = list(), list(), list()
        forget_user_set = set()
        for batch_size in batch_size_list:
            candidate_user_list = list(set(self.user_list) - forget_user_set)
            super_node = random.sample(candidate_user_list, 1)[0]
            emb_super_node_to_mat = [U2M2P[super_node][m] for m in self.mat_list]
            emb_super_node_to_con = [U2C2P[super_node][c] for c in self.con_list]
            other_node_with_sim = list()
            for i in range(len(self.user_list)):
                # SKY RULE : if other_node is obeies SKY RULE, then other_node will not consider.
                other_node = self.user_list[i]
                if other_node == other_node: # SKY RULE
                    if super_node == other_node:
                        forget_user_set.add(super_node)
                    else:
                        emb_other_node_to_mat = [U2M2P[other_node][m] for m in self.mat_list]
                        emb_other_node_to_con = [U2C2P[other_node][c] for c in self.con_list]
                        similarity = self.calculate_similarity(super_emb1=emb_super_node_to_mat, 
                                                            super_emb2=emb_super_node_to_con, 
                                                            other_emb1=emb_other_node_to_mat, 
                                                            other_emb2=emb_other_node_to_con)
                    other_node_with_sim.append([other_node, similarity])
            other_node_with_sim = sorted(other_node_with_sim, reverse=True, key= lambda x: x[1])
            if len(other_node_with_sim) >= batch_size:
                pos_other_node = [element[0] for element in other_node_with_sim[:batch_size]]
                # build token
                U2M2P_token, U2C2P_token, M2U2P_token = dict(), dict(), dict()
                for u in [super_node] + pos_other_node:
                    M2P, C2P = U2M2P[u], U2C2P[u]
                    U2M2P_token[u], U2C2P_token[u] = M2P, C2P
                    for m in self.mat_list:
                        M2U2P_token[m][u] = M2P[m]
                sub_U2M2P_list.append(U2M2P_token)
                sub_U2C2P_list.append(U2C2P_token)
                sub_M2U2P_list.append(M2U2P_token)
                # add forget node to forget_user_set
                forget_user_set = forget_user_set | set(pos_other_node)
            else:
                error = True
        if error is True:
            return sub_U2M2P_list, sub_U2C2P_list, sub_M2U2P_list
        else:
            return None, None, None
        
    def _to_mat(self, sub_U2M2P_list, sub_M2U2P_list):
        sub_U2small_M2P_list = list()
        for i in range(len(sub_U2M2P_list)):
            sub_U2M2P, sub_M2U2P = sub_U2M2P_list[i], sub_M2U2P_list[i]
            mat_with_median = list()
            for m in self.mat_list:
                mat_with_median.append([m, np.median(list(sub_M2U2P[m].values()))])
            mat_with_median = sorted(mat_with_median, reverse=True, key= lambda x: x[1])
            pos_m_list = [element[0] for element in mat_with_median[:mat_with_median]]
            u_list = list(sub_U2M2P.keys())
            sub_U2small_M2P = dict()
            for u in u_list:
                sub_U2small_M2P[u] = dict()
                for m in pos_m_list:
                    sub_U2small_M2P[u][m] = sub_U2M2P[u][m]
            sub_U2small_M2P_list.append(sub_U2small_M2P)
        return sub_U2small_M2P_list




          