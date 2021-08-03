


from tqdm import tqdm
import numpy as np


def add_label_feature_to_rating_data(rating_data):
    # add label to rating_data
    rating_data['label'] = [np.nan for _ in range(rating_data.shape[0])]
    uid_list = list(set(rating_data['client_sn']))
    for uid in tqdm(uid_list):
        dat = rating_data[rating_data['client_sn'] == uid]
        index = dat.index
        score_list = list(dat['material_points'])
        max_score = max(score_list)
        label_list = []
        for score in score_list:
            if score == max_score:
                label_list.append(1)
            else:
                label_list.append(0)
        rating_data.loc[index, 'label']   = label_list
    return rating_data


def remove_repeat_bias_data_in_train_rating_data(train_rating_data, pred_obj_name='mat'):
    # In train data, remove all same score data
    # Remove mat for each high pos_ratio user (?)
    uid_list = list(set(train_rating_data['client_sn']))
    rating_data_wo_repeat = list()
    for uid in tqdm(uid_list):
        dat = train_rating_data[train_rating_data['client_sn'] == uid]
        if pred_obj_name == 'mat':
            if len(set(dat['material_points'])) > 1:
                rating_data_wo_repeat.append(dat)
                score_list = list(dat['label'])
        elif pred_obj_name == 'con':
            if len(set(dat['consultant_points'])) > 1:
                rating_data_wo_repeat.append(dat)
                score_list = list(dat['label'])            
    train_rating_data = pd.concat(rating_data_wo_repeat).reset_index(drop=True)
    return train_rating_data





def transform_date_to_age(date_str, categorical=True):
    if date_str != 'None':
        age_val = 2021 - pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S').year
        if categorical is False:
            return age_val
        else:
            if age_val <= 30:
                return '0~30'
            elif age_val > 30 and age_val < 50:
                return '30~50'
            else:
                return '50~'   
    else:
        return 'None'


def data_process_of_user_data(user_data):
    # data process of user_data
    user_data = user_data.fillna('None')
    user_data['Client_Sex'].replace('N','None')
    user_data['birthday'] = user_data['birthday'].apply(lambda x: transform_date_to_age(x))
    user_data['JobClassName'].replace('Undefined','None')
    user_data['IndustryClassName'].replace('Undefined','None')
    user_data_with_it = user_data[['client_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName','user_interest_tag_list']]
    user_data = user_data[['client_sn','Client_Sex','birthday','education','JobClassName','IndustryClassName']]
    return user_data, user_data_with_it




