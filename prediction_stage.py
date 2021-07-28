
import pandas as pd
from prediction_stage_util.classroom_schedule_algorithm import Classroom_Schedule_Algorithm
from prediction_stage_util.funnel_algorithm import Funnel_Algorithm
from prediction_stage_util.organic_test_data_to_model_input import Organic_Test_Data_TO_Model_Input






class Prediction_Stage:
    def __init__(self, Adult_or_Junior='Adult'):
        # init parameter
        self.save_path = save_path
        # call algorithm objection
        self.classroom_schedule_algorithm_obj = Classroom_Schedule_Algorithm()
        self.funnel_algo_obj = Funnel_Algorithm()


    def build_model_input_from_organic_test_data(self, meta_object, Adult_or_Junior):
        organic_test_data_to_model_input_obj = Organic_Test_Data_TO_Model_Input(meta_object=meta_object, Adult_or_Junior=Adult_or_Junior)
        test_data = organic_test_data_to_model_input_obj.main()
        return test_data 


    def load_dcgs_model_object(self, Adult_or_Junior):
        # load random_forest model and meta dobject
        if Adult_or_Junior == 'Adult':
            rf_model = pickle.load(open(self.save_path + 'rf_AD_model.pickle', 'rb'))
            meta_object = pickle.load(open(self.save_path + 'meta_object_AD.pkl', 'rb'))
        elif Adult_or_Junior == 'Junior':
            rf_model = pickle.load(open(self.save_path + 'rf_Jr_model.pickle', 'rb'))
            meta_object = pickle.load(open(self.save_path + 'meta_object_Jr.pkl', 'rb'))    
        user_num = meta_object['user_num']
        item_num = meta_object['item_num']
        normalized_adj = meta_object['normalized_adj']
        self.mat_individual_col = meta_object['mat_individual_col']
        self.mat_individual_dat = meta_object['mat_individual_dat']
        self.mat_overall_dat = meta_object['mat_overall_dat']
        # laod ngcf model
        if Adult_or_Junior == 'Adult':
            load_model = [True, self.save_path+'ngcf_AD_model.pt']
        elif Adult_or_Junior == 'Junior':
            load_model = [True, self.save_path+'ngcf_Jr_model.pt']
        ngcf_obj = NGCF_Modeling(user_num, item_num, normalized_adj, feature_list, load_model=load_model)
        return rf_model, meta_object, ngcf_obj


    def build_UMP_data(self, test_data, rf_model ,meta_object, ngcf_obj):
        uid2index = meta_object['uid2index']
        mat2index = meta_object['mat2index']
        feature_list = meta_object['feature_list']
        U2M2P= ngcf_obj.recommend(test_data, uid2index ,mat2index, rf_model, feature_list)
        return U2M2P
    

    def main(self, Adult_or_Junior='Adult'):
        # load pre-train model
        rf_model, meta_object, ngcf_obj = self.load_dcgs_model_object(Adult_or_Junior=Adult_or_Junior)
        # data process for model input
        test_data = self.build_model_input_from_organic_test_data(meta_object=meta_object, Adult_or_Junior=Adult_or_Junior)
        # build R
        U2M2P = self.build_UMP_data(test_data=test_data, rf_model=rf_model ,meta_object=meta_object, ngcf_obj=ngcf_obj)
        # funnel part (user)
        sub_U2M2P_list = self.funnel_algo_obj._to_user_by_KMeans(U2M2P)
        # funnel part (mat)
        sub_U2M2P_list = self.funnel_algo_obj._to_mat_by_prob_and_SKY_RULE(sub_U2M2P_list)
        # classroom schedule algo part
        scheduled_classroom = list()
        for sub_U2M2P in sub_U2M2P_list:
            dat = self.classroom_schedule_algorithm_obj.main(sub_U2M2P)
            scheduled_classroom.append(dat)
        scheduled_classroom = pd.concatenate(scheduled_classroom, axis=0)
        return scheduled_classroom

