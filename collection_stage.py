






from collect_stage_util.demo_collect_class_data import Collect_Class_Data_BY_RD_API
from collect_stage_util.demo_collect_mat_data import Collect_Material_From_DimMaterialPool
from collect_stage_util.demo_write_organic_data_to_db import Organic_Data_TO_DB
from load_data.rating_part_BETA import collect_rating_data_func
 

class Collection_Stage:
    def __init__(self, train_start_date='2021-01-01', train_end_date='2021-04-01'):
        self.ccdbra_obj = Collect_Class_Data_BY_RD_API()
        rating_dat = collect_rating_data_func(batch_size=1000, exp_date=[train_start_date,train_end_date], save_path=None,show_dat=True)
        old_mat_list = list(set(rating_dat['MaterialID']))
        #old_mat_list = [10001]
        self.cmfd_obj = Collect_Material_From_DimMaterialPool(old_mat_list)
        self.odtd_obj = Organic_Data_TO_DB(db_name='demo_database', collection_name='demo_api')


    def add_mat_list_to_class_dat_based_on_session_sn(self, class_dat, Adult_or_Jr, go_to_db=False):
        if Adult_or_Jr == 'Adult':
            matType = 'Adult'
        elif Adult_or_Jr == 'Jr':
            matType = 'Junior'
        mat_list = self.cmfd_obj.main(matType=matType) # it will add SKY_RULE in future
        session_sn_list = list(set(class_dat['SessionSn']))
        for s_sn in session_sn_list:
            dat = class_dat[class_dat['SessionSn'] == s_sn]
            u_list = list(dat['ClientSn'])
            a_list = list(dat['Level'])
            u_list_str = ','.join([str(u) for u in u_list])
            a_list_str = ','.join([str(a) for a in a_list])
            m_list_str = ','.join([str(m) for m in mat_list])
            if go_to_db is True:
                self.odtd_obj.write_organic_data_to_db(session_sn=s_sn, 
                                                       client_sn=u_list_str,
                                                       MaterialID=m_list_str, 
                                                       attend_level=a_list_str)
        class_dat_with_mat = None
        return class_dat_with_mat


    def main(self, parameter=dict, service=str):
        if service == 'range_date':
            StartDateTime = parameter['StartDateTime']
            Adult_or_Jr = parameter['Adult_or_Jr']
            class_dat = self.ccdbra_obj.main_for_range_date(StartDateTime=StartDateTime, 
                                                            Adult_or_Jr=Adult_or_Jr)
        elif service == 'client_sn':
            client_sn = parameter['client_sn']
            Adult_or_Jr = parameter['Adult_or_Jr']
            class_dat = self.ccdbra_obj.main_for_client_sn(client_sn=client_sn, 
                                                           Adult_or_Jr=Adult_or_Jr)
        if class_dat is not None:
            self.add_mat_list_to_class_dat_based_on_session_sn(class_dat, Adult_or_Jr, go_to_db=True)




if __name__ == '__main__':
    parameter_for_range_date = \
        {
            "StartDateTime" : "2021-07-21 09:00:00",
            "Adult_or_Jr" : "Adult"
        }
    parameter_for_client_sn = \
        {
            "client_sn" : 20035639,
            "Adult_or_Jr" : "Adult",
        }   
    cs_obj = Collection_Stage()
    cs_obj.main(parameter=parameter_for_range_date, service='range_date')