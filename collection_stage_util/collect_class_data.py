

import requests
import pandas as pd
 
  
 

class Collect_Class_Data_BY_RD_API:
    def __init__(self):
        self.url_for_range_date = 'http://tutorgroupapi.tutorabc.com/ReservationDataAccess/Class/GetClassInformationByStartHourSstNumber'
        self.url_for_session_sn  = 'http://tutorgroupapi.tutorabc.com/ReservationDataAccess/Class/GetClassInformationBySessionSn'
        self.headers = {"Token": 'iGwDivB7W0s%3d'}
    

    def load_organic_data_by_GetClassInformationByStartHourSstNumber_API(self,StartDateTime, BrandIds_list=[int]):
        json_format = \
            {
                "Data": {
                    "StartDateTime": StartDateTime,
                    'BrandIds':BrandIds_list
                },
                "Uuid": "2e34515f-388d-4d22-b77d-cc257d7742d2"
            }
        obj = requests.post(self.url_for_range_date,json=json_format,headers=self.headers)
        return obj
    

    def main(self, StartDateTime="2021-07-21 09:00:00", Adult_or_Junior='Adult'):
        if Adult_or_Junior == 'Adult':
            BrandIds_list = [1]
        elif Adult_or_Junior == 'Junior':
            BrandIds_list = [33]
        obj = self.load_organic_data_by_GetClassInformationByStartHourSstNumber_API(StartDateTime=StartDateTime, BrandIds_list=BrandIds_list)
        dat = pd.DataFrame(dict(obj.json())['Data'])
        dat = dat.fillna('None')
        dat = dat[(dat['LobbySn']=='None') & (dat['SessionSn']!='None')][['ClientSn', 'StartDateTime', 'Level', 'SessionSn']]
        return dat
    
