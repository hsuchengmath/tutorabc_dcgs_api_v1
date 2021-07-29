
import string




def similarity(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    if len(set(s1) | set(s2)) != 0:
        return len(set(s1) & set(s2)) / len(set(s1) | set(s2))
    return 0


def get_all_interest_tag(rating_user_mat_data):
    user_interest_tag_list = list(rating_user_mat_data['user_interest_tag_list'])
    MDCGSID_ENname = list(rating_user_mat_data['MDCGSID_ENname'])
    user_mat_organic_interest_list = user_interest_tag_list + MDCGSID_ENname
    interest_tag_list = set()
    for element in user_mat_organic_interest_list:
        if element != 'None':
            element = element.split('/**/')
            interest_tag_list = interest_tag_list | set(element)
    return list(interest_tag_list)


def build_transform_func(interest_tag_list):
    organic_it2uniform_it = dict()
    # remove punc, set lower
    interest_tag_list_regu = list()
    for it in interest_tag_list:
        it = it.lower().translate(str.maketrans('', '', string.punctuation))
        interest_tag_list_regu.append(it)
    organic_it2uniform_it = dict()
    for i in range(len(interest_tag_list)):
        organic_it2uniform_it[interest_tag_list[i]] = set()
        it_s1 = interest_tag_list_regu[i]
        organic_it2uniform_it[interest_tag_list[i]].add(it_s1)
        for j in range(len(interest_tag_list)):
            it_s2 = interest_tag_list_regu[j]
            sim = similarity(it_s1, it_s2)
            if sim >= 0.4:
                organic_it2uniform_it[interest_tag_list[i]].add(it_s2)
        organic_it2uniform_it[interest_tag_list[i]] = sorted(organic_it2uniform_it[interest_tag_list[i]])[0]
    return organic_it2uniform_it


def calculate_overlap_num(rating_user_mat_data, organic_it2uniform_it):
    overlap_num = list()
    user_interest_tag_list = list(rating_user_mat_data['user_interest_tag_list'])
    MDCGSID_ENname = list(rating_user_mat_data['MDCGSID_ENname'])
    for i in range(len(user_interest_tag_list)):
        if user_interest_tag_list[i] != 'None' and MDCGSID_ENname[i] != 'None':
            UIT = list(set(user_interest_tag_list[i].split('/**/')))
            MIT = list(set(MDCGSID_ENname[i].split('/**/')))
            UIT_uniform = [organic_it2uniform_it[interest_tag] for interest_tag in UIT]
            MIT_uniform = [organic_it2uniform_it[interest_tag] for interest_tag in MIT]
            overlap_num.append(len(set(UIT_uniform) & set(MIT_uniform)))
        else:
            overlap_num.append(0)
    return overlap_num



def overlap_num_func_main(rating_data_with_UF):
    '''
    1. fillna
    2. get all interest tag
    3. build transform func
    4. calculate overlap num from organci list
    5. put overlap num to organic dat
    '''
    rating_user_mat_data = rating_data_with_UF.fillna('None')
    interest_tag_list = get_all_interest_tag(rating_user_mat_data)
    organic_it2uniform_it = build_transform_func(interest_tag_list)
    overlap_num = calculate_overlap_num(rating_user_mat_data, organic_it2uniform_it)
    rating_data_with_UF = rating_data_with_UF.drop(['user_interest_tag_list','MDCGSID_ENname'], axis=1)
    rating_data_with_UF['overlap_num'] = overlap_num
    return rating_data_with_UF