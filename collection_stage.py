

from collection_stage_util.collect_class_data import Collect_Class_Data_BY_RD_API
from collection_stage_util.collect_user_with_history_mat import Collect_User_With_History_Mat
from collection_stage_util.collect_mat_data import Collect_Material_From_DimMaterialPool
from collection_stage_util.write_organic_data_to_db import Organic_Data_TO_DB
from modeling_stage_util.load_data.rating_part_BETA import collect_rating_data_func
 

 

class Collection_Stage:
    def __init__(self, train_start_date='2021-01-01', train_end_date='2021-04-01'):
        self.collect_class_data_by_rd_api_obj = Collect_Class_Data_BY_RD_API()
        rating_dat = collect_rating_data_func(batch_size=1000, exp_date=[train_start_date,train_end_date], save_path=None,show_dat=True)
        old_mat_list = list(set(rating_dat['MaterialID']))
        #old_mat_list = [10001]
        self.organic_data_to_db_obj = Organic_Data_TO_DB(db_name='demo_database', collection_name='demo_api')
        self.collect_user_with_history_mat_obj = Collect_User_With_History_Mat(obj=self.organic_data_to_db_obj)
        self.collect_mat_from_edw_table_obj = Collect_Material_From_DimMaterialPool(old_mat_list)


    def class_data_with_cahdidate_mat_go_to_db(self, class_dat, candidate_mat_list, Adult_or_Junior):
        u_list = list(class_dat['ClientSn'])
        a_list = list(class_dat['Level'])
        u_list_str = ','.join([str(u) for u in u_list])
        a_list_str = ','.join([str(a) for a in a_list])
        m_list_str = ','.join([str(m) for m in candidate_mat_list])
        self.organic_data_to_db_obj.define_collection(collection_name='organic_class_data-'+Adult_or_Junior)
        self.organic_data_to_db_obj.write_class_data_with_mat_to_db(client_sn=u_list_str, MaterialID=m_list_str, attend_level=a_list_str)


    def main(self, parameter=dict):
        StartDateTime = parameter['StartDateTime']
        Adult_or_Junior = parameter['Adult_or_Junior']
        # collect class data by rd-api
        class_dat = self.collect_class_data_by_rd_api_obj.main(StartDateTime=StartDateTime, Adult_or_Junior=Adult_or_Junior)
        if class_dat is not None:
            # build data of SKY-RULE (history mat)
            user2history_mat = self.collect_user_with_history_mat_obj.main(class_dat=class_dat, Adult_or_Junior=Adult_or_Junior)
            # build candidate mat list
            candidate_mat_list = self.collect_mat_from_edw_table_obj.main(user2history_mat=user2history_mat,Adult_or_Junior=Adult_or_Junior)
            # store class_data, candidate data to go to db
            self.class_data_with_cahdidate_mat_go_to_db(class_dat, candidate_mat_list, Adult_or_Junior)
        else:
            print('[ERROR] : cannot collect class_dat by given date.')
            quit()




if __name__ == '__main__':
    parameter = \
        {
            "StartDateTime" : "2021-07-21 09:00:00",
            "Adult_or_Junior" : "Adult"
        }  
    collection_stage_obj = Collection_Stage()
    collection_stage_obj.main(parameter=parameter)