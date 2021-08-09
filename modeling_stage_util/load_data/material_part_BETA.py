





import pandas as pd
from modeling_stage_util.load_data.util import SQL_query
from tqdm import tqdm

 

def find_material_feature_func(mat_id_batch=None, mat_type_batch=None):
    # Material basic feature part (DimMaterialPool)
    sql = '''
        select 
            MaterialID,MTitle, MShared
        from 
            DimMaterialPool
        where 
            MaterialID in ({})
            and MTitle is not NULL and MShared is not NULL and MaterialType = {}
        '''
    mat_id_batch_str = ','.join([str(element) for element in mat_id_batch]) 
    sql = sql.format(mat_id_batch_str, "'"+mat_type_batch+"'")
    dat_basic = SQL_query(query=sql, database='BI_EDW')
    dat_basic = dat_basic.assign(MaterialType =pd.Series([mat_type_batch for _ in range(len(dat_basic['MaterialID']))]))

    # Material interest tag part (DimMatDCGSAdultBridge, DimMatDCGSJuniBridge)
    basic_sql = '''
        select 
             {0}.MaterialID, DimMatDCGS.MDCGSID_ENname
        from 
            {0}
        left join
            DimMatDCGS
        on 
            {0}.MDCGSID = DimMatDCGS.NewDCGSID
        where 
            {0}.MaterialID in ({1})
        '''
    if dat_basic.shape != (0,0) and mat_type_batch == 'Adult':
        sql = basic_sql.format('DimMatDCGSAdultBridge',mat_id_batch_str)
    elif dat_basic.shape != (0,0) and mat_type_batch == 'Junior':
        sql = basic_sql.format('DimMatDCGSJuniBridge',mat_id_batch_str)
    else:
        return None
    dat_interest_tag = SQL_query(query=sql, database='BI_EDW')



    if 'MDCGSID_ENname' in dat_interest_tag.columns:
        MaterialID = list(dat_interest_tag['MaterialID'])
        MDCGSID_ENname = list(dat_interest_tag['MDCGSID_ENname'])
        # builf matid to tag
        matid2IT = dict()
        for i in range(len(MaterialID)):
            if MaterialID[i] not in matid2IT:
                matid2IT[MaterialID[i]] = set()
            matid2IT[MaterialID[i]].add(MDCGSID_ENname[i])
        MaterialID_list = list(matid2IT.keys())
        MDCGSID_ENname_list = list()
        for matid in MaterialID_list:
            IT_list = list(matid2IT[matid])
            IT_list = [element.rstrip() for element in IT_list if element is not None]
            MDCGSID_ENname_list.append('/**/'.join(IT_list))
        dat_interest_tag = {'MaterialID':MaterialID_list, 'MDCGSID_ENname'  : MDCGSID_ENname_list}
        dat_interest_tag = pd.DataFrame(dat_interest_tag)
        dat = dat_basic.merge(dat_interest_tag, on=['MaterialID'],how='left')
    else:
        MDCGSID_ENname = []
        dat = dat_basic.assign(MDCGSID_ENname =pd.Series(MDCGSID_ENname))
    return dat



def collect_mat_data_func(additional_mat=[],additional_mat_type=[],rating_data_path=None,save_path=None,batch_size = 1000):
    # load rating data
    if rating_data_path is None:
        rating = pd.read_csv('rating_BETA.csv')
    else:
        rating = pd.read_csv(rating_data_path) 
    MaterialID_list = list(rating['MaterialID']) + additional_mat
    MaterialType_list = list(rating['MaterialType']) + additional_mat_type
    MaterialID2MaterialType = dict()
    for i,mat_id in enumerate(MaterialID_list):
        if mat_id not in MaterialID2MaterialType:
            MaterialID2MaterialType[mat_id] = set()
        MaterialID2MaterialType[mat_id].add(MaterialType_list[i])
    MaterialID_set = list(MaterialID2MaterialType.keys())
    # collect data by batch MaterialID
    df_material_data = list()
    batch_num = int(len(MaterialID_set) / batch_size) + 1
    for i in tqdm(range(batch_num)):
        mat_id_batch = MaterialID_set[(i*batch_size) : ((i+1)*batch_size)]    
        mat_type_batch = [MaterialID2MaterialType[mat_id] for mat_id in mat_id_batch]
        mat_id_batch_adult, mat_id_batch_jr = list(), list()
        for j, mat_type in enumerate(mat_type_batch):
            mat_id = mat_id_batch[j]
            if 'Adult' in mat_type:
                mat_id_batch_adult.append(mat_id)
            if 'Junior' in mat_type:
                mat_id_batch_jr.append(mat_id)
        if len(mat_id_batch_adult) != 0:
            dat = find_material_feature_func(mat_id_batch=mat_id_batch_adult, mat_type_batch='Adult')
            if dat is not None:
                df_material_data.append(dat)
        if len(mat_id_batch_jr) != 0:
            dat = find_material_feature_func(mat_id_batch=mat_id_batch_jr, mat_type_batch='Junior')
            if dat is not None:
                df_material_data.append(dat)
    df_material_data = pd.concat(df_material_data).reset_index(drop=True)
    if save_path is None:
        df_material_data.to_csv('material_feature_BETA.csv',index=False) 
    else:
        df_material_data.to_csv(save_path,index=False) 

 


if __name__ == '__main__':
    collect_mat_data_func(rating_data_path='data/rating_BETA_Jan.csv',save_path='data/material_feature_BETA_Jan.csv')
