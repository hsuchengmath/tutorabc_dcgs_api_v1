


import pandas as pd
from tqdm import tqdm
from util import SQL_query

 
def collect_review_data_func(rating_data_path=None, save_path=None, batch_size = 5000):
    # load target data
    if rating_data_path is None:
        rating = pd.read_csv('rating_BETA.csv')
    else:
        rating = pd.read_csv(rating_data_path)
    con_sn_list = list(set(rating['con_sn']))
    # collect data by these distinct batch con_sn
    df_con_data = list()
    batch_num = int(len(con_sn_list) / batch_size) + 1
    for i in tqdm(range(batch_num)):
        con_sn_batch = con_sn_list[(i*batch_size) : (i+1)*batch_size]
        if len(con_sn_batch) != 0:
            con_sn_batch_str = ','.join([str(con) for con in con_sn_batch if pd.isna(con) is False])
            sql = '''
            select 
                con_sn, area, con_gender, BirthDay, Complexion,
                Ethnic, Accent, Accent2, NativeLanguage
            from 
                DimConsultant with(nolock)
            where
                con_sn is not NULL
                and con_sn in ({})
            '''
            sql =  sql.format(con_sn_batch_str)
            dat = SQL_query(query=sql, database='BI_EDW')
            if dat.shape[0] !=0:
                df_con_data.append(dat)
    df_con_data = pd.concat(df_con_data).reset_index(drop=True)
    if save_path is None:
        df_con_data.to_csv('consultant_feature_BETA_Jan.csv',index=False)
    else:
        df_con_data.to_csv(save_path,index=False)
    print('Finish collecting consultant data!!')



if __name__ == '__main__':
    collect_review_data_func(rating_data_path='data/rating_BETA_Jan.csv',save_path='data/consultant_feature_BETA_Jan.csv')




