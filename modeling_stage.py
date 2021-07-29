
import os
import numpy as np
import pandas as pd
from object_orient_for_JL.data_process_mat import Data_Process
from object_orient_for_JL.util import transform_date_to_age, train_model, predict_score, random_model_performance, ROC_curve_plot, DET_curve_plot
from object_orient_for_JL.ngcf_model import Adj_Matx_Generator
from object_orient_for_JL.ngcf_model import NGCF_Modeling
import pickle
from load_data.rating_part_BETA import collect_rating_data_func
from load_data.user_part_BETA import collect_user_data_func
from load_data.material_part_BETA import collect_mat_data_func
from load_data.review_part_BETA import collect_review_data_func
import pymongo



class Modeling_Stage:
    def __init__(self):
        '''
        we should receive uid,mat of class_dat
        '''
        self.start_date = '2021-01-01'
        self.train_date = '2021-04-01'
        self.end_date = '2021-06-01'
        self.save_path = 'saved_model/'
        # mongoDB init
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["demo_database"]
        self.mycol_AD = mydb["organic_class_data-"+'Adult']
 
    def go_to_mongoDB_to_get_user_and_mat(self, mycol, Adult_or_Junior='Adult'):
        uid_list, pb_id, mat_list, mat_type = list(), list(), list(), list()
        for x in mycol.find():
            uid_str_list = x['client_sn'].split(',')
            u_list = list(set([int(element) for element in uid_str_list]))
            uid_list += u_list
            mat_str_list = x['MaterialID'].split(',')
            m_list = [int(element) for element in mat_str_list]
            mat_list += m_list
        uid_list = list(set(uid_list))
        if Adult_or_Junior == 'Adult':
            pb_id = [1 for _ in range(len(uid_list))]
        elif Adult_or_Junior == 'Junior':
            pb_id = [33 for _ in range(len(uid_list))]
        mat_list = list(set(mat_list))
        mat_type = [Adult_or_Junior for _ in range(len(mat_list_AD))]
        return uid_list , pb_id, mat_list, mat_type

    def load_train_data(self):
        additional_user , additional_pb_id, additional_mat, additional_mat_type = self.go_to_mongoDB_to_get_user_and_mat(mycol=self.mycol_AD)
        collect_rating_data_func(batch_size=1000, exp_date=[self.start_date,self.end_date], save_path='train_data/rating_BETA_Jan.csv')
        collect_user_data_func(additional_user=additional_user,additional_pb_id=additional_pb_id,rating_data_path='train_data/rating_BETA_Jan.csv',save_path='train_data/user_feature_BETA_Jan.csv')
        collect_mat_data_func(additional_mat=additional_mat,additional_mat_type=additional_mat_type,rating_data_path='train_data/rating_BETA_Jan.csv',save_path='train_data/material_feature_BETA_Jan.csv')
        collect_review_data_func(rating_data_path='train_data/rating_BETA_Jan.csv',save_path='train_data/review_BETA_Jan.csv')

    def data_process(self, load_data=False):
        if load_data is True:
            self.load_train_data()
        data_process_obj = Data_Process(self.start_date, self.train_date, self.end_date)
        data_process_obj.main()
        #
        mat_individual_dat_Jr = data_process_obj.mat_individual_dat_Jr
        mat_overall_dat_Jr = data_process_obj.mat_overall_dat_Jr
        mat_individual_dat_AD = data_process_obj.mat_individual_dat_AD
        mat_overall_dat_AD = data_process_obj.mat_overall_dat_AD
        train_data_AD_wo_na = data_process_obj.train_data_AD_wo_na
        train_data_Jr_wo_na = data_process_obj.train_data_Jr_wo_na
        train_data_AD = data_process_obj.train_data_AD
        train_data_Jr = data_process_obj.train_data_Jr
        label_AD = data_process_obj.label_AD
        label_Jr = data_process_obj.label_Jr
        feature_list = data_process_obj.feature_list
        #
        rf_train_data_package_AD = \
            {
             'train_data' : train_data_AD_wo_na,
             'label' : label_AD, 
             'mat_overall_dat' : mat_overall_dat_AD,
             'mat_individual_dat' : mat_individual_dat_AD,
             'feature_list' : feature_list,
            }
        rf_train_data_package_Jr = \
            {
             'train_data' : train_data_Jr_wo_na,
             'label' : label_Jr,
             'mat_overall_dat' : mat_overall_dat_Jr,
             'mat_individual_dat' ; mat_individual_dat_Jr,
             'feature_list' : feature_list,
            }        

        ngcf_data_process_package = \
            {
             'train_data_AD' : train_data_AD,
             'train_data_Jr' : train_data_Jr
            }
        return rf_train_data_package_AD, rf_train_data_package_Jr , ngcf_data_process_package

    def train_random_forest_model(self, train_data_package, mode='AD', save_model=False):
        train_data = train_data_package['train_data']
        label = train_data_package['label']
        # train RF model
        model_rf = train_model(train_data=train_datatrain_data, label=label_AD)
        if save_model is True:
            # store model
            pickle.dump(model_rf, open(self.save_path + 'rf_'+mode+'_model.pickle', 'wb'))

    def re_index_func(self, train_data):
        # data process of NGCF model 
        client_sn_train = list(train_data['client_sn'])
        mat_train = list(train_data['MaterialID'])
        client_sn = list(set(client_sn_train))
        mat_list = list(set(mat_train))
        uid2index, mat2index, uid_index,mat_index = dict(), dict(),0,0
        for uid in client_sn:
            uid2index[uid] = uid_index
            uid_index +=1
        for mat in mat_list:
            mat2index[mat] = mat_index
            mat_index +=1
        client_sn_train = [uid2index[uid] for uid in client_sn_train]
        mat_train = [mat2index[mat] for mat in mat_train]
        label_train = list(train_data['label'])
        user_num = len(client_sn)
        item_num = len(mat_list)
        return user_num, item_num, client_sn_train, mat_train, label_train, interaction_train, uid2index, mat2index
 
    def build_adj_matrix(self, user_num, item_num, train_uid_dat_with_train_mat_dat):
        amg_obj = Adj_Matx_Generator(user_num, item_num, train_uid_dat_with_train_mat_dat)
        onehot_adj,norm_adj,normalized_adj = amg_obj.main()
        return onehot_adj,norm_adj,normalized_adj

    def build_ngcf_train_data(self, train_data):
        # reindex for NGCF model
        user_num, item_num, \
        client_sn_train, mat_train, label_train,\
        uid2index, mat2index = \
            self.re_index_func(train_data)
        # build adj matrix
        onehot_adj, norm_adj , normalized_adj = \
            self.build_adj_matrix(user_num, item_num, [client_sn_train, mat_train])
        ngcf_train_data_package = \
            {
             'user_num' : user_num,  
             'item_num' : item_num,  
             'adj' : normalized_adj,  
             'client_sn_train' : client_sn_train, 
             'mat_train' : mat_train, 
             'label_train' : label_train,
             'uid2index' : uid2index,
             'mat2index' : mat2index,
            }
        return ngcf_train_data_package

    def train_ngcf_model(self, train_data_package, save_model,mode):
        # put data_package
        user_num = train_data_package['user_num']
        item_num = train_data_package['item_num']
        adj = train_data_package['adj']
        client_sn_train = train_data_package['client_sn_train']
        mat_train = train_data_package['mat_train']
        label_train = train_data_package['label_train']
        save_model_ = [save_model, self.save_path+'ngcf_'+mode+'_model.pt']
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        ngcf_obj = NGCF_Modeling(user_num, item_num,adj, load_model=[False, None]) 
        ngcf_obj.train(client_sn_train, mat_train, label_train, save_model=save_model_)
        return ngcf_obj

    def save_Meta_object(self, ngcf_train_data_package ,rf_train_data_package, mode=None):
        data = \
            {
             'user_num' : ngcf_train_data_package['user_num'], 
             'item_num' : ngcf_train_data_package['item_num'],,
             'uid2index' : ngcf_train_data_package['uid2index'], 
             'mat2index' : ngcf_train_data_package['mat2index'],
             'normalized_adj' : ngcf_train_data_package['adj'], 
             'feature_list' : rf_train_data_package['feature_list'],
             'mat_overall_dat' : rf_train_data_package['mat_overall_dat'], 
             'mat_individual_dat' : rf_train_data_package['mat_individual_dat'],
            }
        with open(self.save_path+'model_other_object_'+ mode +'.pkl', "wb") as f:
            pickle.dump(data, f)

    def main(self):
        # load data and data process
        rf_train_data_package_AD, rf_train_data_package_Jr , ngcf_data_process_package = self.data_process(load_data=True)
        # train random forest model and store model to bucket (AD)
        self.train_random_forest_model(rf_train_data_package=rf_train_data_package_AD,mode='AD', save_model=True)
        # train random forest model and store model to bucket (Jr)
        self.train_random_forest_model(rf_train_data_package=rf_train_data_package_Jr, mode='Jr',save_model=True)
        # build_ngcf_train_data (AD part)
        ngcf_train_data_package_AD = self.build_ngcf_train_data(train_data=ngcf_data_process_package['train_data_AD'])
        # build_ngcf_train_data (Jr part)
        ngcf_train_data_package_Jr = self.build_ngcf_train_data(train_data=ngcf_data_process_package['train_data_Jr'])
        # train model (AD part)
        ngcf_obj_AD = self.train_ngcf_model(ngcf_train_data_package_AD, mode='AD', save_model=True)
        self.save_other_object(ngcf_train_data_package_AD,mode='AD')
        # train model (Jr part)
        ngcf_obj_Jr = self.train_ngcf_model(ngcf_train_data_package_Jr, mode='Jr', save_model=True)
        self.save_other_object(ngcf_train_data_package_Jr,mode='Jr')



modeling_stage_obj = Modeling_Stage()
modeling_stage_obj.main()
