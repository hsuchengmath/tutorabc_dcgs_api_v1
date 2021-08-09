


import random
  

 



class Distributed_Layer:
    def __init__(self, U2M2P, user_list, mat_list, constrain_user_num2user_id):
        self.U2M2P = U2M2P
        self.user_list = user_list
        self.mat_list = mat_list
        self.constrain_user_num2user_id = constrain_user_num2user_id

    def main(self, constrain_num=int):
        constrained_user = self.constrain_user_num2user_id[constrain_num]
        constrained_U2M2P = dict()
        for u in constrained_user:
            M2P= self.U2M2P[u]
            constrained_U2M2P[u] = M2P
        return constrained_U2M2P, constrained_user





def Waterfall_Layer(constrain_user_num2user_id, constrain_num_list):
    constrain_num_list = sorted(constrain_num_list, reverse=True)
    additional_user_id_list = []
    for constrained_num in constrain_num_list:
        user_num = len(constrain_user_num2user_id[constrained_num])
        if user_num % constrained_num != 0:
            additional_num = int(user_num % constrained_num)
            additional_user_id = random.sample(constrain_user_num2user_id[constrained_num], additional_num)
            constrain_user_num2user_id[constrained_num] = \
                list(set(constrain_user_num2user_id[constrained_num]) - set(additional_user_id))
            additional_user_id_list += additional_user_id
        if len(additional_user_id_list) >= constrained_num:
            additional_user_id_list = random.sample(additional_user_id_list, len(additional_user_id_list))
            need_num = len(additional_user_id_list) - int(len(additional_user_id_list) % constrained_num)
            constrain_user_num2user_id[constrained_num] += additional_user_id_list[:need_num]
            additional_user_id_list = additional_user_id_list[need_num:]
    return constrain_user_num2user_id