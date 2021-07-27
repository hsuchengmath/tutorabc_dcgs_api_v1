




import pymongo
 

class Organic_Data_TO_DB:
    def __init__(self, db_name='demo_database', collection_name='demo_api'):
        # open mongoDB in localhost
        self.myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        self.mydb = self.myclient[db_name]
        self.mycol = self.mydb[collection_name]
    def write_organic_data_to_db(self,session_sn, client_sn, MaterialID, attend_level):
        mydict = { 'session_sn': session_sn, 'client_sn':client_sn, 'MaterialID':MaterialID,'attend_level':attend_level }
        x = self.mycol.insert_one(mydict)
