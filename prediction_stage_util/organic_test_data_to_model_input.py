

import pymongo
from object_orient_for_JL.util import transform_date_to_age
from object_orient_for_JL.interest_tag_overlap_num import overlap_num_func_main
import pandas as pd

 
 
 
class Organic_Test_Data_TO_Model_Input:
    def __init__(self, mat_individual_dat,mat_overall_dat, feature_list, user_mode):
        # reload database
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["demo_database"]
        if user_mode == 'Adult':
            self.mycol = mydb["demo_api"]
        # load data
        self.user_data = pd.read_csv('train_data/user_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.mat_data = pd.read_csv('train_data/material_feature_BETA_Jan.csv', encoding='utf-8-sig')
        # init
        self.mat_individual_dat = mat_individual_dat
        self.mat_overall_dat = mat_overall_dat
        self.feature_list = feature_list

    def add_data_to_class_data(self, class_data,s_sn, u_list, m_list, a_list):
        for i in range(len(u_list)):
            for j in range(len(m_list)):
                class_data['session_sn'].append(s_sn)
                class_data['client_sn'].append(u_list[i])
                class_data['MaterialID'].append(m_list[j])
                class_data['attend_level'].append(a_list[i])
        return class_data

    def build_class_data(self, session_sn_list):
        # given session_list, then output matched class_data
        if session_sn_list is not None:
            session_sn_list = set(session_sn_list)
        # init 
        class_data = {'session_sn':[],'client_sn':[],'MaterialID':[],'attend_level':[]}
        # take data from DB
        for x in self.mycol.find():
            s_sn = int(x['session_sn'])
            uid_str_list = x['client_sn'].split(',')
            u_list = [int(element) for element in uid_str_list]
            mat_str_list = x['MaterialID'].split(',')
            m_list = [int(element) for element in mat_str_list]
            al_str_list = x['attend_level'].split(',')
            a_list = [int(element) for element in al_str_list]
            if session_sn_list is None or s_sn in session_sn_list:
                class_data = self.add_data_to_class_data(class_data,s_sn, u_list, m_list, a_list)
        class_data = pd.DataFrame(class_data)
        return class_data

    def build_user_mat_data(self, class_data):
        uid_list = list(class_data['client_sn'])
        attend_level_list = list(class_data['attend_level'])
        U2A = dict()
        for i in range(len(uid_list)):
            U2A[uid_list[i]] = attend_level_list[i]
        uid_list = list(set(uid_list))
        old_mat_list = list(set([token for token in list(self.mat_overall_dat.index)]))
        client_sn_list, MaterialID_list, attend_level_list = [], [], []
        for i in range(len(uid_list)):
            u = uid_list[i]
            a = U2A[u]
            for m in old_mat_list:
                client_sn_list.append(u)
                MaterialID_list.append(m)
                attend_level_list.append(a)
        user_mat_data = {'client_sn':client_sn_list,'MaterialID':MaterialID_list,'attend_level':attend_level_list}
        user_mat_data = pd.DataFrame(user_mat_data)
        return user_mat_data

    def data_process_of_Udat(self):
        user_data = self.user_data.fillna('None')
        user_data['Client_Sex'].replace('N','None')
        user_data['birthday'] = user_data['birthday'].apply(lambda x: transform_date_to_age(x))
        user_data['JobClassName'].replace('Undefined','None')
        user_data['IndustryClassName'].replace('Undefined','None')
        user_data_with_it = user_data[['client_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName','user_interest_tag_list']]
        return user_data_with_it

    def add_overlap_num_feature(self, class_data_with_UF_it, mat_data):
        mat_data = mat_data[['MaterialID', 'MDCGSID_ENname']]
        class_data_with_UF_it = pd.merge(class_data_with_UF_it, mat_data, on=['MaterialID'], how='left')
        class_data_with_UF_it = overlap_num_func_main(class_data_with_UF_it)    
        return class_data_with_UF_it
 
    def main(self, session_sn_list):
        # build class data
        self.class_data = self.build_class_data(session_sn_list=session_sn_list)
        # build user_mat_data
        self.user_mat_data = self.build_user_mat_data(self.class_data)
        # data process for user data
        user_data = self.data_process_of_Udat()
        # merger user_mat_data and user_data
        user_mat_data_with_UF_it = pd.merge(self.user_mat_data, user_data, on=['client_sn'], how='left')
        # add overla_numn feature
        user_mat_data_with_UF_it = self.add_overlap_num_feature(user_mat_data_with_UF_it, self.mat_data)
        # build model input
        dat_individual = pd.merge(user_mat_data_with_UF_it, self.mat_individual_dat, on=['MaterialID','Client_Sex','birthday','education','JobClassName','IndustryClassName'], how='left')
        dat_overall = pd.merge(user_mat_data_with_UF_it, self.mat_overall_dat, on=['MaterialID'], how='left')
        dat = dat_individual.fillna(dat_overall)#.dropna()
        return dat#[self.feature_list]
 