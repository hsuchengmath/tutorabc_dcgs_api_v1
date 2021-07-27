



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

import pandas as pd
import random


 

class Collect_Material_From_DimMaterialPool:
    def __init__(self, old_mat_list):
        self.old_mat_list = old_mat_list
        self.alpha = 0.33
        self.old_mat_len = len(list(set(old_mat_list)))
    
    def load_organic_mat_data(self, matType=None):
        sql = '''
        select 
            MaterialID  
        from 
            DimMaterialPool with (nolock)
        where 
            MaterialType = '{}'
        '''
        sql = sql.format(matType)
        dat = SQL_query(query=sql,database='BI_EDW')
        return dat

    def main(self, matType='Adult'):
        mat_data = self.load_organic_mat_data(matType=matType)
        new_mat_list = random.sample(list(set(mat_data['MaterialID']) - set(self.old_mat_list)), int(self.alpha * self.old_mat_len))
        return self.old_mat_list + new_mat_list
