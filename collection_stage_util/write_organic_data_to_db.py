

 

 
import pymongo
 

class Organic_Data_TO_DB:
    def __init__(self, db_name='demo_database'):
        # open mongoDB in localhost
        self.myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        self.mydb = self.myclient[db_name]
        #self.mycol = self.mydb[collection_name]
    
    def define_collection(self, collection_name):
        self.mycol = self.mydb[collection_name]

    def write_class_data_with_mat_to_db(self, client_sn, MaterialID, attend_level):
        mydict = {'client_sn':client_sn, 'MaterialID':MaterialID,'attend_level':attend_level}
        x = self.mycol.insert_one(mydict)

    def write_SKY_RULE_data_to_db(self, client_sn, history_MaterialID):
        mydict = { 'client_sn':client_sn, 'history_MaterialID':history_MaterialID }
        x = self.mycol.insert_one(mydict)