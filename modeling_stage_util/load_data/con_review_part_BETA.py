


import pandas as pd
from tqdm import tqdm
from modeling_stage_util.load_data.util import SQL_query

 
def collect_review_data_func(rating_data_path=None, save_path=None, batch_size = 5000):
    # load target data
    if rating_data_path is None:
        rating = pd.read_csv('rating_BETA.csv')
    else:
        rating = pd.read_csv(rating_data_path)
    session_sn_list = list(set(rating['session_sn']))
    # collect data by these distinct batch session_sn
    df_review_data = list()
    batch_num = int(len(session_sn_list) / batch_size) + 1
    for i in tqdm(range(batch_num)):
        session_sn_batch = session_sn_list[(i*batch_size) : (i+1)*batch_size]
        if len(session_sn_batch) != 0:
            session_sn_batch_str = ','.join(["'"+str(session)+"'" for session in session_sn_batch])
            sql = ''' 
            select 
                client_sn, con_sn, MaterialID, session_sn,
                complaint_STF, complaint_STS, complaint_UNT, complaint_NFM, complaint_SPO,
                complaint_USW, complaint_NWB, complaint_INT, complaint_NIC, complaint_NCO,
                complaint_NEH, complaint_NPA, complaint_BAT, complaint_NPT, complaint_NEP, 
                compliment_CAR, compliment_WDP, compliment_EXT, compliment_GTA, compliment_EDW,
                compliment_GIN, compliment_VIV, compliment_ENT, compliment_FAS, compliment_FAM,
                compliment_HAR, compliment_PAT, compliment_RIG, consultant_ACL, consultant_ACN,
                consultant_ACD, consultant_PRG, consultant_PRN, consultant_PRB,
                count_consultant_points, C_Point, count_materials_points, M_Point, count_overall_points, T_Point
            from 
                FactSessionEvaluation with(nolock)
            where
                client_sn is not NULL
                and con_sn is not NULL
                and MaterialID is not NULL
                and session_sn in ({})
            '''
            sql =  sql.format(session_sn_batch_str)
            dat = SQL_query(query=sql, database='BI_EDW')
            if dat.shape[0] !=0:
                df_review_data.append(dat)
    df_review_data = pd.concat(df_review_data).reset_index(drop=True)
    if save_path is None:
        df_review_data.to_csv('review_BETA.csv',index=False)
    else:
        df_review_data.to_csv(save_path,index=False)
    print('Finish collecting rating data!!')



if __name__ == '__main__':
    collect_review_data_func(rating_data_path='data/rating_BETA_Jan.csv',save_path='data/con_review_BETA_Jan.csv')
