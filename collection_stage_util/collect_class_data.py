

import requests
import pandas as pd

 


class Collect_Class_Data_BY_RD_API:
    def __init__(self):
        self.url_for_range_date = 'http://tutorgroupapi.tutorabc.com/ReservationDataAccess/Class/GetClassInformationByStartHourSstNumber'
        self.url_for_client_sn = 'http://tutorgroupapi.tutorabc.com/ReservationDataAccess/Class/GetClassInformationByClient'       
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
    
    def load_organic_data_by_GetClassInformationByClient_API(self, ClientSn):
        json_format = \
            {
                "Data": {
                    "ClientSn": ClientSn,
                },
                "Uuid": "2e34515f-388d-4d22-b77d-cc257d7742d2"
            }
        obj = requests.post(self.url_for_client_sn,json=json_format,headers=self.headers)
        return obj

    def load_organic_data_by_GetClassInformationBySessionSn_API(self, SessionSn=str):
        json_format = \
            {
                "Data": {
                    "SessionSns": [str(SessionSn)],
                },
                "Uuid": "2e34515f-388d-4d22-b77d-cc257d7742d2"
            }
        obj = requests.post(self.url_for_session_sn,json=json_format,headers=self.headers)
        return obj

    def main_for_range_date(self, StartDateTime="2021-07-21 09:00:00", Adult_or_Jr='Adult'):
        if Adult_or_Jr == 'Adult':
            BrandIds_list = [1]
        elif Adult_or_Jr == 'Jr':
            BrandIds_list = [33]
        obj = self.load_organic_data_by_GetClassInformationByStartHourSstNumber_API(StartDateTime=StartDateTime, BrandIds_list=BrandIds_list)
        dat = pd.DataFrame(dict(obj.json())['Data'])
        dat = dat.fillna('None')
        dat = dat[(dat['LobbySn']=='None') & (dat['SessionSn']!='None')][['ClientSn', 'StartDateTime', 'Level', 'SessionSn']]
        return dat
    
    def main_for_client_sn(self, client_sn, Adult_or_Jr):

        obj = self.load_organic_data_by_GetClassInformationByClient_API(ClientSn=client_sn)
        dat = pd.DataFrame(dict(obj.json())['Data']).fillna('None')   
        dat = dat[(dat['LobbySn']=='None') & (dat['SessionSn']!='None')]
        if dat.shape[0] != 0:
            matched_session_sn = list(dat.sort_values(by=['StartDateTime'], ascending=False)['SessionSn'])[0]
            obj = self.load_organic_data_by_GetClassInformationBySessionSn_API(SessionSn=matched_session_sn)
            dat = pd.DataFrame(dict(obj.json())['Data'])   
            return dat
        else:
            return None
