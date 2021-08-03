

  
  
 
import pymongo
 

class Organic_Data_TO_DB:
    def __init__(self, db_name='demo_database', db_type='mongoDB'):
        if db_type == 'mongoDB':
            # open mongoDB in localhost
            self.myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
            self.mydb = self.myclient[db_name]
            #self.mycol = self.mydb[collection_name]
        elif db_type == 'MySQL':
            a = 0
    
    def define_collection(self, collection_name): 
        self.mycol = self.mydb[collection_name]

    def write_class_data_with_mat_to_db(self, client_sn, MaterialID,con_sn, attend_level):
        mydict = {'client_sn':client_sn, 'MaterialID':MaterialID,'con_sn':con_sn,'attend_level':attend_level}
        x = self.mycol.insert_one(mydict)

    def write_SKY_RULE_data_to_db(self, client_sn, history_MaterialID):
        mydict = { 'client_sn':client_sn, 'history_MaterialID':history_MaterialID }
        x = self.mycol.insert_one(mydict) 
    
    def write_U2Ent2prob_to_db(self, U2Ent2P, user_list, ent_list, pred_obj_name=None):
        client_sn, ent_sn, prob = list(), list(), list()
        for u in user_list:
            for e in ent_list:
                client_sn.append(str(u))
                ent_sn.append(str(e))
                prob.append(str(U2Ent2P[u][e]))
        client_sn = ','.join(client_sn)
        ent_sn = ','.join(ent_sn)
        prob = ','.join(prob)
        if pred_obj_name == 'mat':
            mydict = { 'client_sn':client_sn, 'Material_ID':ent_sn ,'prob':prob}
        elif pred_obj_name == 'con':
            mydict = { 'client_sn':client_sn, 'con_sn':ent_sn, 'prob':prob}
        x = self.mycol.insert_one(mydict)      
        

    def write_scheduled_classroom_to_db(self, scheduled_classroom):
        a = 0
