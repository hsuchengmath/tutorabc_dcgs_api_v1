


 
import random
import gurobipy as gp
from gurobipy import GRB
from gurobipy import * 
import pandas as pd


class Classroom_Schedule_Layer:
    def __init__(self):
        a = 0
     
    def build_R_dict(self, user_id_list, mat_id_list, potential_con_list, sub_U2M2P, sub_U2C2P):
        R_dict = dict()
        for i in user_id_list:
            R_dict[i] = dict()
            for j in mat_id_list:
                R_dict[i][j] = dict()
                for h in potential_con_list:
                    prob1 = sub_U2M2P[i][j]
                    prob2 = sub_U2M2P[i][h]
                    R_dict[i][j][h] = (prob1 + prob2)/2
        return R_dict

    def build_C_dict_and_D_dict(self, user_id_list, mat_id_list, potential_con_list):
        var_C_dict, var_D_dict = dict(), dict()
        for i in user_id_list:
            var_C_dict[i] = dict()
            for h in potential_con_list:
                var_C_dict[i][h] = \
                    self.model.addVar(vtype=GRB.BINARY, 
                                      name="C_"+str(i)+'_'+str(h))
        for h in potential_con_list:
            var_D_dict[h] = dict()
            for j in mat_id_list:
                var_D_dict[h][j] = \
                    self.model.addVar(vtype=GRB.BINARY, 
                                      name="D_"+str(h)+'_'+str(j))  
        return var_C_dict, var_D_dict
    
    def _init_to_model_(self):
        self.model = gp.Model("mip1")
        self.model.update()
        self.model.setParam('OutputFlag',0)
     
    def build_constrain(self,user_id_list, mat_id_list, potential_con_list, var_C_dict, var_D_dict, potential_con_num):
        # build constrain
        # first term
        for h in potential_con_list:
            self.model.addConstr(quicksum(var_C_dict[i][h] for i in user_id_list) == potential_con_num)
        # second term
        for i in user_id_list:
            self.model.addConstr(quicksum(var_C_dict[i][h] for h in potential_con_list) == 1)
        # third term
        for h in potential_con_list:
            self.model.addConstr(quicksum(var_D_dict[h][j] for j in mat_id_list) == 1)
          
    def main(self, sub_U2M2P, sub_U2C2P, potential_con_num, potential_con_list):
        self._init_to_model_()
        user_id_list = list(sub_U2M2P.keys())
        mat_id_list = list(sub_U2M2P[user_id_list[0]].keys())
        R_dict = \
            self.build_R_dict(user_id_list, mat_id_list, potential_con_list, sub_U2M2P, sub_U2C2P)
        var_C_dict, var_D_dict = \
            self.build_C_dict_and_D_dict(user_id_list, mat_id_list, potential_con_list)
        operation = \
            quicksum( \
                R_dict[i][j][h]*var_C_dict[i][h]*var_D_dict[h][j]  \
                for h in potential_con_list \
                for i in user_id_list  \
                for j in mat_id_list \
        )
        self.model.setObjective(operation, GRB.MAXIMIZE)
        self.build_constrain(user_id_list, mat_id_list, potential_con_list, var_C_dict, var_D_dict, potential_con_num)
        self.model.optimize()
        # 透過屬性objVal顯示最佳解
        #print('Obj: %g' % self.model.objVal)
        #print('%s %g' % (v.varName, v.x))
        con2user_list, con2mat = dict(), dict()
        for v in self.model.getVars():
            name = v.varName
            binary = int(v.x)
            if binary == 1:
                if name.split('_')[0] == 'C':
                    user = int(name.split('_')[1])
                    con = int(name.split('_')[2])
                    if con not in con2user_list:
                        con2user_list[con] = list()
                    con2user_list[con].append(user)
                        
                elif name.split('_')[0] == 'D':
                    con = int(name.split('_')[1])
                    mat = int(name.split('_')[2])     
                    if con not in con2mat:
                        con2mat[con] = mat
        SUCM_dat = {'session_sn':[],'client_sn':[],'con_sn':[],'Material_ID':[]}
        s_sn = 0
        for con in potential_con_list:
            user_list, mat = con2user_list[con], con2mat[con]
            for u in user_list:
                SUCM_dat['session_sn'].append(s_sn)
                SUCM_dat['client_sn'].append(u)
                SUCM_dat['con_sn'].append(con)
                SUCM_dat['Material_ID'].append(mat)
            s_sn +=1
        SUCM_dat = pd.DataFrame(SUCM_dat)
        return SUCM_dat
                

        