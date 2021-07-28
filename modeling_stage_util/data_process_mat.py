

import pandas as pd
from tqdm import tqdm
import numpy as np
from object_orient_for_JL.util import transform_date_to_age


class Data_Process:
    def __init__(self, start_date='2021-01-01', train_date='2021-04-01', end_date='2021-05-01'):
        self.start_date = start_date
        self.train_date = train_date
        self.end_date = end_date

    def load_data(self):
        # load data
        self.rating_data = pd.read_csv('train_data/rating_BETA_Jan.csv', encoding='utf-8-sig')
        self.user_data = pd.read_csv('train_data/user_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.mat_data = pd.read_csv('train_data/material_feature_BETA_Jan.csv', encoding='utf-8-sig')
        self.review_data = pd.read_csv('train_data/review_BETA_Jan.csv', encoding='utf-8-sig')
    
    def remove_all_same_score_data(self, rating_data):
        # only materialpointsCNT == 1 in rating_data
        # w/o repeat mat score in rating_data
        rating_data = rating_data[rating_data['materialpointsCNT'] == 1]
        uid_list = list(set(rating_data['client_sn']))
        rating_data_wo_repeat = list()
        for uid in tqdm(uid_list):
            dat = rating_data[rating_data['client_sn'] == uid]
            if len(set(dat['material_points'])) > 1:
                rating_data_wo_repeat.append(dat)
        rating_data = pd.concat(rating_data_wo_repeat).reset_index(drop=True) 
        return rating_data
    
    def add_label_and_tidy_up_rating_data(self, rating_data):
        # add label feature to rating_data
        rating_data['label'] = [np.nan for _ in range(rating_data.shape[0])]
        uid_list = list(set(rating_data['client_sn']))
        for uid in tqdm(uid_list):
            dat = rating_data[rating_data['client_sn'] == uid]
            index = dat.index
            score_list = list(dat['material_points'])
            max_score = max(score_list)
            label_list = []
            for score in score_list:
                if score == max_score:
                    label_list.append(1)
                else:
                    label_list.append(0)
            rating_data.loc[index, 'label']   = label_list
        # tidy up rating_data
        rating_data = rating_data[['client_sn','MaterialID','session_sn','PurchaseBrandID','attend_level','attend_date','label']]
        return rating_data

    def build_rating_review_data(self, rating_data, review_data):
        # build rating_review_data by merging rating_data, review_data. where key = ['client_sn','MaterialID','session_sn'] (left join)
        review_data = review_data[list(set(review_data.columns)-{'favor_status_after','collect_status_after'})]
        rating_review_data = pd.merge(rating_data, review_data, on=['client_sn','MaterialID','session_sn'], how='left')
        return rating_review_data, review_data

    def fillna_in_user_data_and_distinguish_it(self, user_data):
        user_data = user_data.fillna('None')
        user_data['Client_Sex'].replace('N','None')
        user_data['birthday'] = user_data['birthday'].apply(lambda x: transform_date_to_age(x))
        user_data['JobClassName'].replace('Undefined','None')
        user_data['IndustryClassName'].replace('Undefined','None')
        user_data_with_it = user_data[['client_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName','user_interest_tag_list']]
        user_data = user_data[['client_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName']]
        return user_data, user_data_with_it

    def add_user_data_to_rating_data_and_rating_review_data(self, user_data,user_data_with_it, rating_review_data, rating_data):
        rating_review_data_with_UF = pd.merge(rating_review_data, user_data, on=['client_sn'], how='left')
        rating_data_with_UF = pd.merge(rating_data, user_data_with_it, on=['client_sn'], how='left')
        mat_data = self.mat_data[['MaterialID', 'MDCGSID_ENname']]
        rating_data_with_UF = pd.merge(rating_data_with_UF, mat_data, on=['MaterialID'], how='left')
        return rating_review_data_with_UF, rating_data_with_UF

    def build_interaction_feature_and_add_to_rating_data_UF(self, rating_review_data_with_UF, rating_data_with_UF):
        mat_individual_col = list(set(rating_review_data_with_UF.columns)-{'client_sn','MaterialID','session_sn','PurchaseBrandID','attend_level','material_points','con_sn','label','attend_date','Client_Sex','birthday','education','JobClassName','IndustryClassName'})
        mat_individual_dat = rating_review_data_with_UF.groupby(['MaterialID','Client_Sex','birthday','education','JobClassName','IndustryClassName']).mean()[mat_individual_col]
        mat_overall_dat = rating_review_data_with_UF.groupby(['MaterialID']).mean()[mat_individual_col]
        # build rating_matF_data by merging rating_data, mat_individual_dat
        rating_individual_matF_data = pd.merge(rating_data_with_UF, mat_individual_dat, on=['MaterialID','Client_Sex','birthday','education','JobClassName','IndustryClassName'], how='left')
        rating_overall_matF_data = pd.merge(rating_data_with_UF, mat_overall_dat, on=['MaterialID'], how='left')
        rating_matF_data = rating_individual_matF_data.fillna(rating_overall_matF_data)
        return mat_individual_col, mat_individual_dat, mat_overall_dat, rating_matF_data

    def distinguish_AD_Jr_data(self, rating_matF_data):
        # sperate adult, jr data by PurchaseBrandID
        rating_matF_data_AD = rating_matF_data[rating_matF_data['PurchaseBrandID']==1]
        rating_matF_data_Jr = rating_matF_data[rating_matF_data['PurchaseBrandID']!=1]
        rating_matF_data_AD = rating_matF_data_AD[list(set(rating_matF_data_AD.columns)-{'session_sn','PurchaseBrandID','Client_Sex','birthday','education','JobClassName','IndustryClassName'})]
        rating_matF_data_Jr = rating_matF_data_Jr[list(set(rating_matF_data_Jr.columns)-{'session_sn','PurchaseBrandID','Client_Sex','birthday','education','JobClassName','IndustryClassName'})]
        return rating_matF_data_AD, rating_matF_data_Jr

    def split_into_train_test_data(self, rating_matF_data_AD, rating_matF_data_Jr):
        # sperate train, test data by attend_date
        train_data_AD = rating_matF_data_AD[(rating_matF_data_AD['attend_date'] >= self.start_date) & (rating_matF_data_AD['attend_date'] < self.train_date)]
        test_data_AD = rating_matF_data_AD[(rating_matF_data_AD['attend_date'] >= self.train_date) & (rating_matF_data_AD['attend_date'] < self.end_date)]
        train_data_Jr = rating_matF_data_Jr[(rating_matF_data_Jr['attend_date'] >= self.start_date) & (rating_matF_data_Jr['attend_date'] < self.train_date)]
        test_data_Jr = rating_matF_data_Jr[(rating_matF_data_Jr['attend_date'] >= self.train_date) & (rating_matF_data_Jr['attend_date'] < self.end_date)]
        train_data_AD = train_data_AD[list(set(train_data_AD.columns)-{'attend_date'})]
        test_data_AD = test_data_AD[list(set(test_data_AD.columns)-{'attend_date'})]
        train_data_Jr = train_data_Jr[list(set(train_data_Jr.columns)-{'attend_date'})]
        test_data_Jr = test_data_Jr[list(set(test_data_Jr.columns)-{'attend_date'})]
        train_data_AD, test_data_AD, train_data_Jr, test_data_Jr = self.train_test_dropna(train_data_AD, test_data_AD, train_data_Jr, test_data_Jr)
        return train_data_AD, test_data_AD, train_data_Jr, test_data_Jr

    def train_test_dropna(self, train_data_AD, test_data_AD, train_data_Jr, test_data_Jr):
        train_data_AD = train_data_AD.dropna()
        test_data_AD = test_data_AD.dropna()
        train_data_Jr = train_data_Jr.dropna()
        test_data_Jr = test_data_Jr.dropna()
        return train_data_AD, test_data_AD, train_data_Jr, test_data_Jr

    def build_ground_true_data_and_remove_label_in_data(self, train_data_AD, test_data_AD, train_data_Jr, test_data_Jr):
        label_AD = np.array(train_data_AD['label'])
        label_Jr = np.array(train_data_Jr['label'])
        train_data_AD_wo_label = train_data_AD[list(set(train_data_AD.columns)-{'label'})]
        train_data_Jr_wo_label = train_data_Jr[list(set(train_data_Jr.columns)-{'label'})]
        ground_truth_AD = np.array(test_data_AD['label'])
        ground_truth_Jr = np.array(test_data_Jr['label'])
        test_data_AD_wo_label = test_data_AD[list(set(test_data_AD.columns)-{'label'})]
        test_data_Jr_wo_label = test_data_Jr[list(set(test_data_Jr.columns)-{'label'})]
        return label_AD, label_Jr, ground_truth_AD, ground_truth_Jr, train_data_AD_wo_label, train_data_Jr_wo_label, test_data_AD_wo_label, test_data_Jr_wo_label

    def add_overlap_num_feature(self, rating_data_with_UF):
        from object_orient_for_JL.interest_tag_overlap_num import overlap_num_func_main
        rating_data_with_UF = overlap_num_func_main(rating_data_with_UF)
        return rating_data_with_UF
    
    def main(self):
        self.load_data()
        self.rating_data = self.remove_all_same_score_data(self.rating_data)
        self.rating_data = self.add_label_and_tidy_up_rating_data(self.rating_data)
        self.rating_review_data, self.review_data = self.build_rating_review_data(self.rating_data, self.review_data)
        self.user_data, self.user_data_with_it = self.fillna_in_user_data_and_distinguish_it(self.user_data)
        self.rating_review_data_with_UF, self.rating_data_with_UF = self.add_user_data_to_rating_data_and_rating_review_data(self.user_data,self.user_data_with_it, self.rating_review_data, self.rating_data)
        self.rating_data_with_UF = self.add_overlap_num_feature(self.rating_data_with_UF)
        self.mat_individual_col, self.mat_individual_dat,self.mat_overall_dat, self.rating_matF_data = self.build_interaction_feature_and_add_to_rating_data_UF(self.rating_review_data_with_UF, self.rating_data_with_UF)
        self.rating_matF_data_AD, self.rating_matF_data_Jr = self.distinguish_AD_Jr_data(self.rating_matF_data)
        self.train_data_AD, self.test_data_AD, self.train_data_Jr, self.test_data_Jr = self.split_into_train_test_data(self.rating_matF_data_AD, self.rating_matF_data_Jr)
        self.label_AD, self.label_Jr, self.ground_truth_AD, self.ground_truth_Jr, self.train_data_AD_wo_label, self.train_data_Jr_wo_label, self.test_data_AD_wo_label, self.test_data_Jr_wo_label = \
            self.build_ground_true_data_and_remove_label_in_data(self.train_data_AD, self.test_data_AD, self.train_data_Jr, self.test_data_Jr)
        self.feature_list = list(set(self.train_data_AD_wo_label.columns) - {'client_sn','MaterialID'})
        # return self.label_AD, self.label_Jr, self.ground_truth_AD, self.ground_truth_Jr, self.train_data_AD_wo_label, self.train_data_Jr_wo_label, self.test_data_AD_wo_label, self.test_data_Jr_wo_label
        # return self.train_data_AD, self.test_data_AD, self.train_data_Jr, self.test_data_Jr
        # retutrn self.mat_individual_col