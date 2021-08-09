



import numpy as np
import pandas as pd
from tqdm import tqdm
#from load_data.util_BETA import SQL_query, find_user_interest_tag, build_E2E_func
from modeling_stage_util.load_data.util import SQL_query, find_user_interest_tag, build_E2E_func
 
def build_E2E(client_sn=None, PurchaseBrandID=None):
    # jobID -> jobname
    sql = '''
    select JobID,JobName,JobClassName  from DimJob with (nolock) 
    '''
    dat = SQL_query(query=sql, database='BI_EDW')
    JobID = list(dat['JobID'])
    JobName = list(dat['JobName'])
    JobClassName = list(dat['JobClassName'])
    JobID2name = build_E2E_func(ent1_list=JobID, ent2_list=JobName, mode='job')
    JobID2classname = build_E2E_func(ent1_list=JobID, ent2_list=JobClassName, mode='job')
    # industryID -> industryname
    sql = '''
    select IndustryID,IndustryName,IndustryClassName from DimIndustry with (nolock) 
    '''
    dat = SQL_query(query=sql, database='BI_EDW')
    IndustryID = list(dat['IndustryID'])
    IndustryName = list(dat['IndustryName'])
    IndustryClassName = list(dat['IndustryClassName'])
    IndustryID2name = build_E2E_func(ent1_list=IndustryID, ent2_list=IndustryName, mode='industry')
    IndustryID2classname = build_E2E_func(ent1_list=IndustryID, ent2_list=IndustryClassName, mode='industry')
    # eduCode -> eduName
    sql ='''
    select  DISTINCT eduCode, eduName from DimEducation
    '''
    dat = SQL_query(query=sql, database='BI_EDW')
    eduCode = list(dat['eduCode'])
    eduName = list(dat['eduName'])
    eduCode2eduName = build_E2E_func(ent1_list=eduCode, ent2_list=eduName, mode='education')
    # client_sn -> PurchaseBrandID
    client_sn2PurchaseBrandID = \
        build_E2E_func(ent1_list=client_sn, ent2_list=PurchaseBrandID, mode='clsn->purchase-brand')
    return JobID2name, JobID2classname, IndustryID2name, IndustryID2classname, eduCode2eduName, client_sn2PurchaseBrandID


def find_user_feature_func(uid_list=None, E2E_list=None):
    ## user basic feature
    sql = '''
    select 
        client_sn,Client_Sex,birthday,Client_JobID,Client_IndustryID,education  
    from 
        DimClient with (nolock) 
    where 
        DimClient.CountryName = N'臺灣' 
        and DBSource = 'muchnewdb'
        and  DimClient.client_sn in {}
    order by
        client_sn
    '''
    uid_list = sorted(uid_list)
    uid_list_str = '('+','.join([str(uid) for uid in uid_list])+')'
    sql = sql.format(uid_list_str)
    dat_basic = SQL_query(query=sql, database='BI_EDW')
    # mapping
    dat_basic['education'] = dat_basic['education'].map(E2E_list[0])
    dat_basic = dat_basic.assign(JobName =dat_basic['Client_JobID'].map(E2E_list[1]))
    dat_basic = dat_basic.assign(JobClassName =dat_basic['Client_JobID'].map(E2E_list[2]))
    dat_basic = dat_basic.assign(IndustryName =dat_basic['Client_IndustryID'].map(E2E_list[3]))
    dat_basic = dat_basic.assign(IndustryClassName =dat_basic['Client_IndustryID'].map(E2E_list[4]))
    # add element of query none 
    query_get_none = set(uid_list) - set(dat_basic['client_sn'])
    none_data = {'client_sn':list(query_get_none)}
    none_df = pd.DataFrame(none_data)
    if len(query_get_none) != 0:
        dat_basic = pd.concat([dat_basic, none_df], sort=False)
    dat_basic = dat_basic.sort_values(by='client_sn')
    dat = dat_basic
    ## user interest tag
    # sperate two flow based on adult, jr (user interest part)
    uid_adult, uid_jr = list(), list()
    for uid in uid_list:
        pb_code = E2E_list[5][uid]
        if pb_code == 1:
            uid_adult.append(uid)
        elif pb_code == 3 or pb_code == 33:
            uid_jr.append(uid)
        else:
            print('[ERROR]')
    # get interest | adult : 1
    uid2interest_tag_adult = find_user_interest_tag(uid_adult,mode='adult')
    # get interest | jr : 3 or 33
    uid2interest_tag_jr = find_user_interest_tag(uid_jr,mode='jr')    
    # combine the two flow
    interest_tag_list = []
    for uid in uid_list:
        if uid in uid2interest_tag_adult:
            if len(uid2interest_tag_adult) != 0:
                interest_tag_list.append(uid2interest_tag_adult[uid])
            else:
                interest_tag_list.append(np.nan)
        elif uid in uid2interest_tag_jr:
            if len(uid2interest_tag_jr) != 0:
                interest_tag_list.append(uid2interest_tag_jr[uid])
            else:
                interest_tag_list.append(np.nan)
        else:
            interest_tag_list.append(np.nan)
    dat = dat.assign(user_interest_tag_list = pd.Series(interest_tag_list))
    return dat


def collect_user_data_func(additional_user=[],additional_pb_id=[],rating_data_path=None, save_path=None):
    # load target data
    if rating_data_path is None:
        rating = pd.read_csv('rating_BETA.csv')
    else:
        rating = pd.read_csv(rating_data_path)
    client_sn = list(rating['client_sn']) + additional_user
    PurchaseBrandID = list(rating['PurchaseBrandID']) + additional_pb_id
    # build E2E 
    JobID2name, JobID2classname, IndustryID2name, IndustryID2classname, eduCode2eduName, client_sn2PurchaseBrandID = \
        build_E2E(client_sn=client_sn, PurchaseBrandID=PurchaseBrandID)
    E2E_list = [eduCode2eduName, JobID2name, JobID2classname, IndustryID2name, IndustryID2classname, client_sn2PurchaseBrandID]
    # main
    df_user_data = list()
    batch_size = 5000
    client_sn_set = list(set(client_sn))
    batch_num = int(len(client_sn_set)/batch_size)+1
    for i in tqdm(range(batch_num)):
        client_sn_batch = client_sn_set[(i*batch_size) : (i+1)*batch_size]
        if len(client_sn_batch) > 0:
            df = find_user_feature_func(client_sn_batch,E2E_list)
            df_user_data.append(df)
    df_user_data = pd.concat(df_user_data).reset_index(drop=True)
    if save_path is None:
        df_user_data.to_csv('user_feature_BETA.csv',index=False)
    else:
        df_user_data.to_csv(save_path,index=False)





if __name__ == '__main__':
    collect_user_data_func(rating_data_path='train_data/rating_BETA_Jan.csv',save_path='train_data/user_feature_BETA_Jan.csv')





