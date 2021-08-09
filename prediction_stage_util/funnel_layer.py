
import random
import numpy as np

 




class Greedy_Based_Funnel_Layer:
    def __init__(self, constrained_U2M2P_list, constrain_num_list, mat_list):
        self.constrained_U2M2P_list = constrained_U2M2P_list
        self.constrain_num_list = constrain_num_list
        self.mat_list = mat_list
        self.variable_limit = 2400
        self.expect_subgroup_num = 3


    def _to_user(self):
        constrained_subgroupU2M2P_list, subgroup_constrained_num_list = list(), list()
        for i in range(len(self.constrained_U2M2P_list)):
            constrained_U2M2P = self.constrained_U2M2P_list[i]
            constrain_num = self.constrain_num_list[i]
            user_list = list(constrained_U2M2P.keys())
            forget_user_set = set()
            while True:
                candidate_user_list = list(set(user_list) - forget_user_set)
                if len(candidate_user_list) == 0:
                    break
                super_node = random.sample(candidate_user_list, 1)[0]
                emb_super_node = [constrained_U2M2P[super_node][m] for m in self.mat_list]
                other_node_with_sim = list()
                for i in range(len(candidate_user_list)):
                    # SKY RULE : if other_node is obeies SKY RULE, then other_node will not consider.
                    other_node = candidate_user_list[i]
                    if other_node == other_node: # SKY RULE
                        if super_node == other_node:
                            forget_user_set.add(super_node)
                        else:
                            emb_other_node = [constrained_U2M2P[other_node][m] for m in self.mat_list]
                            similarity = self.calculate_similarity(super_emb=emb_super_node, 
                                                                   other_emb=emb_other_node)
                        other_node_with_sim.append([other_node, similarity])
                other_node_with_sim = sorted(other_node_with_sim, reverse=True, key= lambda x: x[1])
                pos_other_node = [element[0] for element in other_node_with_sim[:self.expect_subgroup_num * constrain_num]]  
                # add forget node to forget_user_set
                forget_user_set = forget_user_set | set(pos_other_node)
                # build subgroupU2M2P
                subgroupU2M2P = dict()
                for u in [super_node] + pos_other_node:
                    subgroupU2M2P[u] = dict()
                    for m in self.mat_list:
                        subgroupU2M2P[u][m] = constrained_U2M2P[u][m]   
                constrained_subgroupU2M2P_list.append(subgroupU2M2P) 
                subgroup_constrained_num_list.append(constrain_num)
        return constrained_subgroupU2M2P_list, subgroup_constrained_num_list



    def calculate_similarity(self,super_emb, other_emb):
        super_emb = np.array(super_emb)
        other_emb = np.array(other_emb)
        mse = ((super_emb - other_emb)**2).mean(axis=0)     
        similarity = (-1) * mse
        return similarity


    def transport_for_dict(self, A2B2val):
        B2A2val = dict()
        A_list = list(A2B2val.keys())
        B_list = list(A2B2val[A_list[0]].keys())
        for b in B_list:
            B2A2val[b] = dict()
            for a in A_list:
                B2A2val[b][a] = A2B2val[a][b]
        return B2A2val


    def _to_mat(self, subgroup_constrained_U2M2P_list, subgroup_constrained_num_list):
        # transport subgroup_constrained_U2M2P_list
        subgroup_constrained_M2U2P_list = self.transport_for_dict(subgroup_constrained_U2M2P_list)
        # build subR_list
        subR_list = list()
        for i in range(len(subgroup_constrained_M2U2P_list)):
            #subgroup_U2M2P
            subgroup_U2M2P = subgroup_constrained_U2M2P_list[i]
            subgroup_M2U2P = subgroup_constrained_M2U2P_list[i]
            subgroup_constrained_num = subgroup_constrained_num_list[i]
            mat_with_median = list()
            for m in self.mat_list:
                mat_with_median.append([m, np.median(list(subgroup_M2U2P[m].values()))])
            mat_with_median = sorted(mat_with_median, reverse=True, key= lambda x: x[1])
            # determine funnel_mat_len
            u_list = list(subgroup_U2M2P.keys())
            funnel_mat_len = int(self.variable_limit / (len(u_list) * subgroup_constrained_num))
            pos_m_list = [element[0] for element in mat_with_median[:funnel_mat_len]]
            subR = dict()
            for u in u_list:
                subR[u] = dict()
                for m in pos_m_list:
                    subR[u][m] = subgroup_U2M2P[u][m]
            subR_list.append(subR)
        return subR_list


    def main(self, constrained_U2M2P_list, constrain_num_list):
        subgroup_constrained_U2M2P_list, subgroup_constrained_num_list = self._to_user()
        subR_list = self._to_mat(subgroup_constrained_U2M2P_list, subgroup_constrained_num_list)
        return subR_list, subgroup_constrained_num_list
 
          