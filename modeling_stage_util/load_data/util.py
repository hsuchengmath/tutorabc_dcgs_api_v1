

import numpy as np





def build_E2E_func(ent1_list, ent2_list, mode=None):
    Ent1_TO_Ent2 = dict()
    if mode == 'job' or mode == 'industry':
        for i, ent1 in enumerate(ent1_list):
            Ent1_TO_Ent2[ent1] = ent2_list[i]
    elif mode == 'education':
        for i, ent1 in enumerate(ent1_list):
            ent2 = ent2_list[i]
            if ent1 not in Ent1_TO_Ent2:
                Ent1_TO_Ent2[ent1] = list()
            Ent1_TO_Ent2[ent1].append(ent2)
        Ent1_set = list(Ent1_TO_Ent2.keys())
        for ent in Ent1_set:
            Ent1_TO_Ent2[ent] = '/**/'.join(Ent1_TO_Ent2[ent])
    elif mode == 'clsn->purchase-brand':
        for i, ent1 in enumerate(ent1_list):
            ent2 = ent2_list[i]
            if ent1 in Ent1_TO_Ent2:
                Ent1_TO_Ent2[ent1] = min([Ent1_TO_Ent2[ent1],ent2])
            else:
                Ent1_TO_Ent2[ent1] = int(ent2)
    return Ent1_TO_Ent2


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
    pd.set_option('display.max_rows', None)
    return dat

 

def find_user_interest_tag(u_id_list, mode=None):
    if mode == 'adult':
        # adult
        sql = '''
        select
            DimClientDCGSAdultBridge.client_sn, DimMatDCGS.MDCGSID_ENname
        from 
            DimClientDCGSAdultBridge
        left join
            DimMatDCGS
        on 
            DimClientDCGSAdultBridge.ClientDCGSID = DimMatDCGS.NewDCGSID
        where
            DimClientDCGSAdultBridge.client_sn in {}
        '''
        u_id_list_str = '('+','.join([str(element) for element in u_id_list])+')'
        sql = sql.format(u_id_list_str)
        dat = SQL_query(query=sql, database='BI_EDW')
    else:
        # 3&33 jr
        sql = '''
        select
            DimClientDCGSJuniBridge.client_sn ,DimMatDCGS.MDCGSID_ENname
        from 
            DimClientDCGSJuniBridge
        left join
            DimMatDCGS
        on 
            DimClientDCGSJuniBridge.ClientDCGSID = DimMatDCGS.NewDCGSID
        where
            DimClientDCGSJuniBridge.client_sn in {}
        '''
        u_id_list_str = '('+','.join([str(element) for element in u_id_list])+')'
        sql = sql.format(u_id_list_str)
        dat = SQL_query(query=sql, database='BI_EDW')
    uid2interest_tag = dict()
    if len(dat.columns) != 0:
        for clsn in u_id_list:
            MDCGSID_ENname = list(dat[(dat['client_sn']==int(clsn))]['MDCGSID_ENname'])
            MDCGSID_ENname = [element.rstrip() for element in MDCGSID_ENname if isinstance(element, str) is True]
            MDCGSID_ENname = list(set(MDCGSID_ENname))
            if len(MDCGSID_ENname) != 0:
                MDCGSID_ENname = '/**/'.join(MDCGSID_ENname)
            else:
                MDCGSID_ENname = np.nan
            if clsn not in uid2interest_tag:
                uid2interest_tag[int(clsn)] = MDCGSID_ENname
    return uid2interest_tag
