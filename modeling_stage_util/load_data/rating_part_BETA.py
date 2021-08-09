



import pandas as pd
from modeling_stage_util.load_data.util import SQL_query, find_user_interest_tag
from tqdm import tqdm



def collect_rating_data_func(batch_size=1000 ,exp_date=['2021-05-01','2021-06-01'], save_path=None, show_dat=False):
    print('Start collecting rating data!!')
    # collect distinct clinet_sn
    sql = '''
    select
         DISTINCT client_sn
    from 
        FactClassInfoDetail with (nolock)
    where
        attend_date >= {} and attend_date <= {} and PurchaseBrandID in (1,3,33)
    '''
    sql =  sql.format("'"+exp_date[0]+"'", "'"+exp_date[1]+"'")
    dat = SQL_query(query=sql, database='BI_EDW')
    client_sn_list = list(set(dat['client_sn']))
    
    # collect data by these distinct batch client_sn
    df_rating_data = list()
    batch_num = int(len(client_sn_list) / batch_size) + 1
    for i in tqdm(range(batch_num)):
        client_sn_batch = client_sn_list[(i * batch_size) : ((i+1) * batch_size)]
        if len(client_sn_batch) != 0:
            client_sn_batch_str = ','.join([str(uid) for uid in client_sn_batch])
            sql = '''
                select  
                        client_sn, MaterialID,con_sn,session_sn,
                        MaterialType, PurchaseBrandID, attend_level, SchedulingID,
                        attend_date, sestime, week,
                        materialpointsCNT,material_points, consultantpointsCNT, consultant_points
                from 
                    FactClassInfoDetail with (nolock)
                where 
                    attend_date >= {} and attend_date <= {}
                    and client_sn is not NULL and MaterialID is not NULL and PurchaseBrandID is not NULL and session_sn is not NULL
                    and PurchaseBrandID in (1,3,33)
                    and client_sn in ({})
                    and account_type = N'正常使用客戶'
                    and HCatgId = 'EN'
                order by
                    attend_date
                '''
            sql =  sql.format("'"+exp_date[0]+"'", "'"+exp_date[1]+"'", client_sn_batch_str)
            dat = SQL_query(query=sql, database='BI_EDW')
            df_rating_data.append(dat)
    df_rating_data = pd.concat(df_rating_data).reset_index(drop=True)
    if save_path is not None:
        df_rating_data.to_csv(save_path,index=False)
    print('Finish collecting rating data!!')
    if show_dat is True:
        return df_rating_data

    #SchedulingID : SC001 ~ SC006
 
if __name__ == '__main__':
    collect_rating_data_func(batch_size=1000, exp_date=['2021-01-01','2021-06-01'], save_path='data/rating_BETA_Jan.csv')
