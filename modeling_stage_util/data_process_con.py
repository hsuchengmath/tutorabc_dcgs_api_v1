




from modeling_stage_util.util_for_data_process import add_label_feature_to_rating_data
from modeling_stage_util.util_for_data_process import remove_repeat_bias_data_in_train_rating_data
from modeling_stage_util.util_for_data_process import data_process_of_user_data
from modeling_stage_util.add_overlap_num_feature import overlap_num_func_main
import numpy as np
import pandas as pd



class Data_Process_for_con: 
    def __init__(self):
        self.start_date = '2021-01-01'
        self.train_date = '2021-04-01'
        self.end_date = '2021-05-01'

    def load_data(self):
        self.rating_data = pd.read_csv('data/rating_BETA_Jan.csv', encoding='utf-8-sig')
        self.user_data = pd.read_csv('data/user_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.mat_data = pd.read_csv('data/material_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.con_data = pd.read_csv('data/consultant_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.review_data = pd.read_csv('data/con_review_BETA_Jan.csv', encoding='utf-8-sig')

    def data_process_of_rating_data(self, rating_data):
        # filter 1~6 classroom by SchedulingID
        classroom_type = {'SC001','SC002','SC003','SC004','SC005','SC006'}
        rating_data = rating_data[rating_data['SchedulingID'].isin(classroom_type)]
        # select existed data score
        rating_data = rating_data[rating_data['consultantpointsCNT'] == 1]
        # add label feature
        rating_data = add_label_feature_to_rating_data(rating_data=rating_data)
        return rating_data

    def split_train_test_by_attend_date(self, rating_data):
        # split train/test data by attend_date
        train_rating_data = rating_data[(rating_data['attend_date'] >= self.start_date) & (rating_data['attend_date'] < self.train_date)]
        test_rating_data = rating_data[(rating_data['attend_date'] >= self.train_date) & (rating_data['attend_date'] < self.end_date)]
        return train_rating_data, test_rating_data


    def split_AD_Jr_by_PurchaseBrandID(self, train_rating_data, test_rating_data):
        # distinguish AD/Jr by PurchaseBrandID in train, test data
        train_rating_data_AD = train_rating_data[train_rating_data['PurchaseBrandID']==1]
        test_rating_data_AD = test_rating_data[test_rating_data['PurchaseBrandID']==1]
        train_rating_data_Jr = train_rating_data[train_rating_data['PurchaseBrandID']!=1]
        test_rating_data_Jr = test_rating_data[test_rating_data['PurchaseBrandID']!=1]
        return train_rating_data_AD, test_rating_data_AD, train_rating_data_Jr, test_rating_data_Jr

    def data_process_of_FSE_mat_oi(self, train_rating_data, review_data, user_data):
        # oi = overlap / individual
        # build rating_review_data for FSE_mat_oi
        rating_review_data = pd.merge(train_rating_data, review_data, on=['client_sn','con_sn','session_sn'], how='left')
        rating_review_data_with_UF = pd.merge(rating_review_data, user_data, on=['client_sn'], how='left')
        # build FSE_mat_oi
        mat_individual_col = list(set(rating_review_data_with_UF.columns)-{'client_sn','con_sn','session_sn','PurchaseBrandID','attend_level','material_points','con_sn','label','attend_date','Client_Sex','birthday','education','JobClassName','IndustryClassName'})
        mat_individual_dat = rating_review_data_with_UF.groupby(['con_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName']).mean()[mat_individual_col]
        mat_overall_dat = rating_review_data_with_UF.groupby(['con_sn']).mean()[mat_individual_col]
        return mat_individual_dat, mat_overall_dat


    def data_process_of_add_overlap_num_feature(self, train_test_rating_data, mat_data, user_data_with_it, Adult_or_Junior):
        rating_data_with_UF = pd.merge(train_test_rating_data, user_data_with_it, on=['client_sn'], how='left')
        mat_data = mat_data[mat_data['MaterialType']==Adult_or_Junior]
        mat_data = mat_data[['MaterialID', 'MDCGSID_ENname']]
        rating_data_with_UF = pd.merge(rating_data_with_UF, mat_data, on=['MaterialID'], how='left')
        rating_data_with_UF= overlap_num_func_main(rating_data_with_UF)
        return rating_data_with_UF



    def data_process_of_train_rating_data(self, train_test_rating_data, mat_individual_dat, mat_overall_dat, Adult_or_Junior):
        train_test_rating_data = train_test_rating_data[['client_sn','con_sn','session_sn','PurchaseBrandID','attend_level','attend_date','label']]
        mat_individual_dat, mat_overall_dat = self.data_process_of_FSE_con_oi(train_test_rating_data, self.review_data, self.user_data)
        rating_data_with_UF = self.data_process_of_add_overlap_num_feature(train_test_rating_data, self.mat_data, self.user_data_with_it, Adult_or_Junior)
        rating_matF_data = pd.merge(rating_data_with_UF, mat_individual_dat, on=['con_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName'], how='left')
        train_test_data = rating_matF_data[list(set(rating_matF_data.columns)-{'session_sn','PurchaseBrandID','Client_Sex','birthday','education','JobClassName','IndustryClassName','attend_date'})]
        return train_test_data, mat_individual_dat, mat_overall_dat



    def main(self):
        self.load_data()

        self.rating_data = self.data_process_of_rating_data(rating_data=self.rating_data)

        train_rating_data, test_rating_data = self.split_train_test_by_attend_date(rating_data=self.rating_data)

        train_rating_data = remove_repeat_bias_data_in_train_rating_data(train_rating_data=train_rating_data, pred_obj_name='con')

        train_rating_data_AD, test_rating_data_AD, train_rating_data_Jr, test_rating_data_Jr = \
            self.split_AD_Jr_by_PurchaseBrandID(train_rating_data=train_rating_data, test_rating_data=test_rating_data)

        self.user_data, self.user_data_with_it = data_process_of_user_data(self.user_data)

        self.train_data_AD, self.con_individual_dat_AD, self.con_overall_dat_AD = \
            self.data_process_of_train_rating_data(train_rating_data=train_rating_data_AD, 
                                                        mat_individual_dat=None,
                                                        mat_overall_dat=None,
                                                        Adult_or_Junior='Adult')

        self.train_data_Jr, self.con_individual_dat_Jr, self.con_overall_dat_Jr = \
            self.data_process_of_train_rating_data(train_rating_data=train_rating_data_Jr, 
                                                        mat_individual_dat=None,
                                                        mat_overall_dat=None,
                                                        Adult_or_Junior='Junior')

        self.feature_list = list(set(self.train_data_AD.columns)-{'label'})
        self.label_AD = np.array(self.train_data_AD['label'])
        self.train_data_AD_wo_na = self.train_data_AD[self.feature_list]
        self.label_Jr = np.array(self.train_data_Jr['label'])
        self.train_data_Jr_wo_na = self.train_data_Jr[self.feature_list]
