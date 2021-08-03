
import pandas as pd
from prediction_stage_util.classroom_schedule_layer import Classroom_Schedule_Layer
from prediction_stage_util.funnel_layer import Greedy_Based_Funnel_Layer
from prediction_stage_util.organic_test_data_to_model_input import Organic_Test_Data_TO_Model_Input
from modeling_stage_util.ngcf_model import NGCF_Modeling
from collection_stage_util.write_organic_data_to_db import Organic_Data_TO_DB
import pickle

 

 
class Prediction_Stage:
    def __init__(self, Adult_or_Junior='Adult'):
        # init parameter
        self.save_path = 'bucket_model/'
        # Classroom Schedule System
        self.classroom_schedule_layer_obj = Classroom_Schedule_Layer()
        self.funnel_layer_obj = Greedy_Based_Funnel_Layer()
        # go to DB
        self.organic_data_to_MongoDB_obj = Organic_Data_TO_DB(db_name='demo_database',db_type='mongoDB')
        self.organic_data_to_MySQL_obj = Organic_Data_TO_DB(db_name='demo_database',db_type='MySQL')


    def build_model_input_from_organic_test_data(self, meta_object, Adult_or_Junior, pred_obj_name):
        organic_test_data_to_model_input_obj = Organic_Test_Data_TO_Model_Input(meta_object=meta_object, Adult_or_Junior=Adult_or_Junior)
        test_data = organic_test_data_to_model_input_obj.main()
        return test_data 


    def load_dcgs_model_object(self, Adult_or_Junior):
        # load random_forest model and meta dobject
        if Adult_or_Junior == 'Adult':
            # mat part
            rf_model_mat = pickle.load(open(self.save_path + 'rf_mat_AD_model.pickle', 'rb'))
            meta_object_mat = pickle.load(open(self.save_path + 'meta_object_mat_AD.pkl', 'rb'))
            # con part
            rf_model_con = pickle.load(open(self.save_path + 'rf_con_AD_model.pickle', 'rb'))
            meta_object_con = pickle.load(open(self.save_path + 'meta_object_con_AD.pkl', 'rb'))
        elif Adult_or_Junior == 'Junior':
            # mat part
            rf_model_mat = pickle.load(open(self.save_path + 'rf_mat_Jr_model.pickle', 'rb'))
            meta_object_mat = pickle.load(open(self.save_path + 'meta_object_mat_Jr.pkl', 'rb'))   
            # con part
            rf_model_con = pickle.load(open(self.save_path + 'rf_mat_Jr_model.pickle', 'rb'))
            meta_object_con = pickle.load(open(self.save_path + 'meta_object_mat_Jr.pkl', 'rb'))  
        user_num_for_mat, user_num_for_con = meta_object_mat['user_num'], meta_object_con['user_num']
        mat_num, con_num = meta_object_mat['mat_num'], meta_object_con['con_num']
        normalized_adj_for_mat, normalized_adj_for_con = meta_object_mat['normalized_adj'], meta_object_con['normalized_adj']
        feature_list_for_mat, feature_list_for_con = meta_object_mat['feature_list'], meta_object_con['feature_list']
        self.mat_individual_dat, self.con_individual_dat = meta_object_mat['mat_individual_dat'], meta_object_con['con_individual_dat']
        self.mat_overall_dat, self.con_overall_dat = meta_object_mat['mat_overall_dat'], meta_object_con['con_overall_dat']
        # laod ngcf model
        if Adult_or_Junior == 'Adult':
            load_model_mat = [True, self.save_path+'ngcf_mat_AD_model.pt']
            load_model_con = [True, self.save_path+'ngcf_con_AD_model.pt']
        elif Adult_or_Junior == 'Junior':
            load_model_mat = [True, self.save_path+'ngcf_mat_Jr_model.pt']
            load_model_con = [True, self.save_path+'ngcf_con_Jr_model.pt']
        ngcf_obj_mat = NGCF_Modeling(user_num_for_mat, mat_num, normalized_adj_for_mat, load_model=load_model_mat)
        ngcf_obj_con = NGCF_Modeling(user_num_for_con, con_num, normalized_adj_for_con, load_model=load_model_con)
        return rf_model_mat,rf_model_con, meta_object_mat,meta_object_con, ngcf_obj_mat, ngcf_obj_con


    def build_U2Pred_Obj2P_data(self, test_data, rf_model ,meta_object, ngcf_obj, pred_obj_name):
        uid2index = meta_object['uid2index']
        if pred_obj_name == 'mat':
            pred_obj2index = meta_object['mat2index']
        elif red_obj_name == 'con':
            pred_obj2index = meta_object['con2index']
        feature_list = meta_object['feature_list']
        U2pred_obj2P= ngcf_obj.recommend(test_data, uid2index ,pred_obj2index, rf_model, feature_list)
        user_list = list(U2pred_obj2P.keys())
        pred_obj_list = list(U2pred_obj2P[user_list[0]].keys())
        return U2pred_obj2P, user_list, pred_obj_list

 
    def main(self, constrain_user_num2user_id=None, Adult_or_Junior='Adult'):
        # load pre-train model
        rf_model_mat,rf_model_con, meta_object_mat,meta_object_con, ngcf_obj_mat, ngcf_obj_con = \
            self.load_dcgs_model_object(Adult_or_Junior=Adult_or_Junior)
        # data process for model input (mat part)
        test_data_mat = self.build_model_input_from_organic_test_data(meta_object=meta_object_mat, 
                                                                      Adult_or_Junior=Adult_or_Junior, 
                                                                      pred_obj_name='mat')
        # data process for model input (con part)
        test_data_con = self.build_model_input_from_organic_test_data(meta_object=meta_object_con, 
                                                                      Adult_or_Junior=Adult_or_Junior, 
                                                                      pred_obj_name='con')
        # build R (R=U2M2P)
        U2M2P, user_list, mat_list = self.build_U2Pred_Obj2P_data(test_data=test_data_mat, 
                                                                  rf_model=rf_model_mat ,
                                                                  meta_object=meta_object_mat, 
                                                                  ngcf_obj=ngcf_obj_mat, 
                                                                  pred_obj_name='mat')
        # build H (H=U2C2P)
        U2C2P, _, con_list = self.build_U2Pred_Obj2P_data(test_data=test_data_con, 
                                                          rf_model=rf_model_con ,
                                                          meta_object=meta_object_con, 
                                                          ngcf_obj=ngcf_obj_mat, 
                                                          pred_obj_name='con')
        # store U2M2P, U2C2P to mongoDB (~8/3)
        self.organic_data_to_MongoDB_obj.define_collection(collection_name='U2M2P')
        self.organic_data_to_MongoDB_obj.write_U2Ent2prob_to_db(U2Ent2P=U2M2P, user_list=user_list, ent_list=mat_list, pred_obj_name='mat')
        self.organic_data_to_MongoDB_obj.define_collection(collection_name='U2C2P')
        self.organic_data_to_MongoDB_obj.write_U2Ent2prob_to_db(U2Ent2P=U2C2P, user_list=user_list, ent_list=con_list, pred_obj_name='con')
        # build scheduled_classroom
        scheduled_classroom = list()
        distributed_layer_obj = Distributed_Layer(U2M2P, U2C2P, user_list, mat_list, con_list, constrain_user_num2user_id)
        for constrain_num in [1,2,3,4,5,'free']:
            while constrain_num in [1,2,3,4,5,'free','additional']:
                distributed_layer_obj.main(constrain_num=constrain_num)
                distributed_user_list = distributed_layer_obj.distributed_user_list
                distributed_U2M2P = distributed_layer_obj.distributed_U2M2P
                distributed_U2C2P = distributed_layer_obj.distributed_U2C2P
                # funnel layer init
                self.funnel_layer_obj = Greedy_Based_Funnel_Layer(user_list=distributed_user_list, mat_list=mat_list, con_list=con_list)
                # funnel part (user)
                sub_U2M2P_list, sub_U2C2P_list, sub_M2U2P_list = self.funnel_layer_obj._to_user(U2M2P=distributed_U2M2P, U2C2P=distributed_U2C2P)
                if sub_U2M2P_list is None or sub_U2C2P_list is None or sub_M2U2P_list is None:
                    constrain_num = 'additional'
                else:
                    constrain_num = 'non_additional'
            # funnel part (mat)
            sub_U2M2P_list = self.funnel_layer_obj._to_mat(sub_U2M2P_list, sub_M2U2P_list)
            # classroom schedule layer part
            old_con = list()
            for sub_U2M2P in sub_U2M2P_list:
                dat = self.classroom_schedule_layer_obj.main(sub_U2M2P)
                old_con.append(self.classroom_schedule_layer_obj.used_con)
                scheduled_classroom.append(dat)
            distributed_layer_obj.recycle_con(old_con)
        scheduled_classroom = pd.concatenate(scheduled_classroom, axis=0)
        # store scheduled_classroom to MySQL (~8/3)
        self.organic_data_to_MySQL_obj.write_scheduled_classroom_to_db(scheduled_classroom=scheduled_classroom)
        return scheduled_classroom

  


constrain_user_num2user_id = \
    {
        1 : [],
        2 : [],
        3 : [],
        4 : [],
        5 : []
    }