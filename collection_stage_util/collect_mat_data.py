
from tqdm import tqdm

 
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
            and MatCat = 'regular session'
        '''
        sql = sql.format(matType)
        dat = SQL_query(query=sql,database='BI_EDW')
        return dat

    def calculate_weight_for_new_mat(self, new_mat_list, user2history_mat):
        user_list = list(user2history_mat.keys())
        weight_of_new_mat_list = list()
        for i in range(len(new_mat_list)):
            weight = 0    
            mat = new_mat_list[i]
            for uid in user_list:
                if mat in list(set(user2history_mat[uid])):
                    weight +=1
            weight_of_new_mat_list.append(weight)
        return weight_of_new_mat_list

    def main(self,user2history_mat=None, Adult_or_Junior='Adult'):
        matType = Adult_or_Junior
        mat_dat = self.load_organic_mat_data(matType=matType)
        new_mat_list = list(set(mat_dat['MaterialID']) - set(self.old_mat_list))
        new_mat_sampled_num = int(self.alpha * self.old_mat_len)
        sampled_mat_list = list()
        for _ in tqdm(range(new_mat_sampled_num)):
            weight_of_new_mat_list = self.calculate_weight_for_new_mat(new_mat_list, user2history_mat)
            sampled_mat = random.choices(new_mat_list, weights=weight_of_new_mat_list, k=1)[0]
            new_mat_list = list(set(new_mat_list) - {sampled_mat})
            sampled_mat_list.append(sampled_mat)
        return self.old_mat_list + new_mat_list
  