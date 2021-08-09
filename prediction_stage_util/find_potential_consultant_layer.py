
import gurobipy as gp
from gurobipy import GRB
from gurobipy import * 





class Find_Potential_Consultant_Layer:
    def __init__(self, U2C2P, con_list, salary_list, salary_bar):
        self.U2C2P = U2C2P
        self.con_list = con_list
        self.salary_list = salary_list
        self.salary_bar = salary_bar


    def _init_to_model_(self):
        self.model = gp.Model("mip1")
        self.model.update()


    def build_BH_constant_matrix_hashmap(self, subR_list):
        BH_subRindex2con = dict()
        for i,subR in enumerate(subR_list):
            user_in_subR = list(subR.keys())
            BH_subRindex2con[i] = dict()
            for con in self.con_list:
                BH_subRindex2con[i][con] = 0
                for u in user_in_subR:
                    BH_subRindex2con[i][con] += U2C2P[u][con]   
        return BH_subRindex2con


    def build_C_varaible_materix_hashmap(self, subR_list):
        # build var
        var_C_dict = dict()
        for h in range(len(subR_list)):
            var_C_dict[h] = dict()
            for k in self.con_list:
                var_C_dict[h][k] = \
                    self.model.addVar(vtype=GRB.BINARY, name="C_"+str(h)+'_'+str(k))
        return var_C_dict


    def build_objective_function(self, BH_subRindex2con, var_C_dict):
        operation = \
            quicksum(BH_subRindex2con[h][k] * var_C_dict[h][k] for h in range(len(subR_list)) for k in con_list)
        self.model.setObjective(operation, GRB.MAXIMIZE)


    def build_constrain_function(self, var_C_dict, subR_list, required_con_num_list):
        # build constrain
        # first term
        for k in self.con_list:
            self.model.addConstr(quicksum(var_C_dict[h][k] for h in range(len(subR_list))) <=1)
        # second term
        self.model.addConstr(quicksum(var_C_dict[h][k] * self.salary_list[k] for h in range(len(subR_list)) for k in range(len(self.con_list))) <=self.salary_bar)
        # third term
        for h in range(len(subR_list)):
            self.model.addConstr(quicksum(var_C_dict[h][k] for k in self.con_list) == required_con_num_list[h])


    def optimize_model_and_build_C_hashmap(self):
        # opt
        self.model.optimize()
        # build C hashmap
        subR_index2con_list = dict()
        for v in self.model.getVars():
            sub_index = int(v.varName.split('_')[1])
            con = int(v.varName.split('_')[2])
            if sub_index not in subR_index2con_list:
                subR_index2con_list[sub_index] = list()
            if int(v.x) == 1:
                subR_index2con_list[sub_index].append(con)
        return subR_index2con_list


    def build_subC_list_and_potential_con_list(self, subR_list, subR_index2con_list):
        subC_list, potential_con_list = list(), list()
        for i in range(len(subR_list)):
            subR = subR_list[i]
            con_list = subR_index2con_list[i]
            user_list = list(subR.keys())
            subC = dict()
            for u in user_list:
                subC[u] = dict()
                for c in con_list:
                    subC[u][c] = self.U2C2P[u][c]
            subC_list.append(subC)
            potential_con_list.append(con_list)
        return subC_list, potential_con_list
                




    def main(self, subR_list, subgroup_constrained_num_list):
        self._init_to_model_()

        BH_subRindex2con = self.build_BH_constant_matrix_hashmap(subR_list=subR_list)

        var_C_dict = self.build_C_varaible_materix_hashmap(subR_list=subR_list)

        self.build_objective_function(BH_subRindex2con, var_C_dict)

        self.build_constrain_function(var_C_dict, subR_list, required_con_num_list=subgroup_constrained_num_list)

        subR_index2con_list = self.optimize_model_and_build_C_hashmap()

        subC_list, potential_con_list = self.build_subC_list_and_potential_con_list(subR_list, subR_index2con_list)

        return subC_list, potential_con_list

