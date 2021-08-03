


import random


class Distributed_Layer:
    def __init__(self, U2M2P, U2C2P, user_list, mat_list, con_list, constrain_user_num2user_id):
        self.constrain_user_num2user_id = constrain_user_num2user_id
        self.free_user_id = user_list
        self.mat_list = mat_list
        self.con_list = con_list
        self.U2M2P = U2M2P
        self.U2C2P = U2C2P

    def build_distributed_U2M2P_and_U2C2P(self, user_id):
        distributed_U2M2P, distributed_U2C2P = dict(), dict()
        for u in user_id:
            if u not in distributed_U2M2P:
                distributed_U2M2P[u] = dict()
            for m in self.mat_list:
                distributed_U2M2P[u][m] = self.U2M2P[u][m]
            for c in self.con_list:
                distributed_U2C2P[u][c] = self.U2C2P[u][m]
        return distributed_U2M2P, distributed_U2C2P


    def main(self, constrain_num):
        # init
        if constrain_num != 'additional':
            self.distributed_user_list = list()
            self.distributed_U2M2P, self.distributed_U2C2P = dict(), dict()
        if isinstance(constrain_num, int) is True:
            # get non_free_people
            non_free_user_id = self.constrain_user_num2user_id[constrain_num]
            # re-gain free_people
            self.free_user_id = list(set(self.free_user_id) - set(non_free_user_id))
            # if non_free_people is not enough, free_people will give need_num people to non_free_people
            if len(non_free_user_id) % constrain_num != 0:
                need_num = len(non_free_user_id) - int(len(non_free_user_id) % constrain_num)
                # these added_people cannot break SKY RULE (U-U)
                added_user_id = random.sample(self.free_user_id, need_num)
                # re-gain free_people
                self.free_user_id = list(set(self.free_user_id) - set(added_user_id))
                self.distributed_user_list = non_free_user_id + added_user_id
                # build distributed_U2M2P, distributed_U2C2P
                self.distributed_U2M2P, self.distributed_U2C2P = \
                        self.build_distributed_U2M2P_and_U2C2P(self.distributed_user_list)
            else:
                self.distributed_user_list = non_free_user_id
        else:
            if constrain_num == 'additional':
                # get non_free_people
                non_free_user_id = self.distributed_user_list
                # these added_people cannot break SKY RULE (U-U)
                added_user_id = random.sample(self.free_user_id, 1)
                # re-gain free_people
                self.free_user_id = list(set(self.free_user_id) - set(added_user_id))
                self.distributed_user_list = non_free_user_id + added_user_id
                # build distributed_U2M2P, distributed_U2C2P
                self.distributed_U2M2P, self.distributed_U2C2P = \
                        self.build_distributed_U2M2P_and_U2C2P(self.distributed_user_list)
            elif constrain_num == 'free':
                # build distributed_U2M2P, distributed_U2C2P
                self.distributed_U2M2P, self.distributed_U2C2P = \
                        self.build_distributed_U2M2P_and_U2C2P(self.free_user_id)


