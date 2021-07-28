
from collection_stage_util.write_organic_data_to_db import Organic_Data_TO_DB


 
def SQL_query(query=None, database=None):
    import pymssql
    import pandas as pd
    if database == 'marketing':
        conn = pymssql.connect(host='172.16.22.60', 
                               user='tutorabc\\albert_chsu', 
                               password='11122200', 
                               database ='marketing',
                               appname='AI_Professional_Test')
    elif database == 'BI_EDW':
        conn = pymssql.connect(server='172.16.22.60', 
                       user='ai_py_acct', 
                       password='kC^pqB$c2u8p', 
                       database='BI_EDW',
                       appname='AI_Professional_Test')
    elif database == 'scrm_mongodb':
        conn = pymssql.connect(server='172.16.22.60', 
                       user='ai_py_acct', 
                       password='kC^pqB$c2u8p', 
                       database ='scrm_mongodb',
                       appname='AI_Professional_Test')
    cursor = conn.cursor(as_dict=True)
    cursor.execute(query)
    rs = cursor.fetchall()
    dat = pd.DataFrame(rs)
    pd.set_option('display.max_columns', None)
    #print(dat.head())
    return dat




class Collect_User_With_History_Mat:
    def __init__(self, obj):
        self.organic_data_to_db_obj = obj
        self.organic_data_to_db_obj.define_collection(collection_name='SKY_RULE')
    
    def main(self, class_dat, Adult_or_Junior, write_to_db=True):
        user_id_list = list(set(class_dat['ClientSn']))
        MaterialType = Adult_or_Junior
        user2history_mat = dict()
        for uid in user_id_list:
            sql = '''
            select 
                MaterialID
            from
                FactClassInfoDetail with (nolock)
            where
                client_sn = {}
                and MaterialType = '{}'
                and account_type = N'正常使用客戶'
                and HCatgId = 'EN'
            '''
            sql = sql.format(str(user_id_list[i]), MaterialType)
            dat = SQL_query(query=sql, database='BI_EDW')
            history_mat = list(set(dat['MaterialID']))
            user2history_mat[uid] = history_mat
            if write_to_db is True:
                history_MaterialID = ','.join([str(mat) for mat in history_mat])
                self.organic_data_to_db_obj.write_SKY_RULE_data_to_db(client_sn=str(uid), history_MaterialID=history_MaterialID)
        return user2history_mat


