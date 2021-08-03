


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
    elif database == 'cbs_tw':
        conn = pymssql.connect(server='172.16.22.60', 
                       user='tutorabc\\albert_chsu', 
                       password='111222000', 
                       database ='cbs_tw',
                       appname='AI_Professional_Test')
    cursor = conn.cursor(as_dict=True)
    cursor.execute(query)
    rs = cursor.fetchall()
    dat = pd.DataFrame(rs)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #print(dat.head())
    return dat



import pandas as pd


def get_candidate_consultant_by_time(date=str, hour=int):
    sql = '''
    select
        con_fix_schedule.con_sn, con_fix_schedule.expt_work_date, con_fix_schedule_detail.hour, con_fix_schedule_detail.minute
    from
        con_fix_schedule with (nolock)
    join
        con_fix_schedule_detail with (nolock)
    on
        con_fix_schedule.id = con_fix_schedule_detail.schedule_id
    where
        week = '{}'
        and hour = {}
    '''
    temp = pd.Timestamp(date)
    week = str(temp.dayofweek + 1)
    sql = sql.format(week, hour)
    dat = SQL_query(query=sql, database='cbs_tw')
    return list(set(dat['con_sn']))