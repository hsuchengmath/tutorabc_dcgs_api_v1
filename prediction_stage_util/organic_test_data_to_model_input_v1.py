



class Organic_Test_Data_TO_Model_Input:
    def __init__(self, meta_object, Adult_or_Junior, pred_obj_name):
        # go to mongoDB
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        self.mydb = myclient['organic_class_data-'+Adult_or_Junior]
        # meta data and data_process_obj
        self.feature_list = meta_object['feature_list']
        if pred_obj_name == 'mat':
            self.pred_obj_overall_dat = meta_object['mat_overall_dat']
            self.pred_obj_individual_dat = meta_object['mat_individual_dat']
            self.data_process_obj = Data_Process_for_mat()
        elif pred_obj_name == 'con':
            self.pred_obj_overall_dat = meta_object['con_overall_dat']
            self.pred_obj_individual_dat = meta_object['con_individual_dat']
            self.data_process_obj = Data_Process_for_con()

    def add_data_to_UA_pred_obj_dat(self, UA_pred_obj_dat, u_list, ent_list, a_list, ent_name):
        for i in range(len(u_list)):
            for j in range(len(ent_list)):
                UA_pred_obj_dat['client_sn'].append(u_list[i])
                UA_pred_obj_dat[ent_name].append(ent_list[j])
                UA_pred_obj_dat['attend_level'].append(a_list[i])
        return UA_pred_obj_dat


    def build_UA_pred_obj_dat(self, pred_obj_name):
        # init 
        if pred_obj_name == 'mat':
            UA_pred_obj_dat = {'client_sn':[],'MaterialID':[],'attend_level':[]}
            ent_name = 'MaterialID'
        elif pred_obj_name == 'con':
            UA_pred_obj_dat = {'client_sn':[],'con_sn':[],'attend_level':[]}
            ent_name = 'con_sn'
        # take data from DB
        for x in self.mycol.find():
            uid_str_list = x['client_sn'].split(',')
            u_list = [int(element) for element in uid_str_list]
            mat_str_list = x[ent_name].split(',')
            ent_list = [int(element) for element in mat_str_list]
            al_str_list = x['attend_level'].split(',')
            a_list = [int(element) for element in al_str_list]
            UA_pred_obj_dat = self.add_data_to_UA_pred_obj_dat(UA_pred_obj_dat, u_list, ent_list, a_list, ent_name)
        UA_pred_obj_dat = pd.DataFrame(UA_pred_obj_dat)
        return UA_pred_obj_dat

    def main(self):
        UA_pred_obj_dat = self.build_UA_pred_obj_dat(pred_obj_name=self.pred_obj_name)
        test_data, _,_ = \
            self.data_process_obj.data_process_of_train_rating_data(train_rating_data=UA_pred_obj_dat, 
                                                                    individual_dat=self.pred_obj_individual_dat,
                                                                    overall_dat=self.pred_obj_overall_dat,
                                                                    Adult_or_Junior=self.Adult_or_Junior)
        test_data = test_data[self.feature_list]
        return test_data

