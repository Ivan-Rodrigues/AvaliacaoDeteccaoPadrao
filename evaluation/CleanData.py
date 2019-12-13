from datetime import datetime
from random import randint
import numpy as np
import pandas as pd

uids = 'u00','u01','u02', 'u03', 'u04','u05','u08', 'u07', 'u08', 'u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',\
       'u23','u27','u30','u31','u32','u33','u34','u35','u36', 'u41','u42','u43','u44','u45','u46', 'u47', 'u49', 'u50', 'u51','u52','u54','u57','u58','u59'

#uids = 'u00','u02', 'u04','u05','u08',  'u08', 'u09','u10','u12','u13','u14','u16','u17','u19',\
 #      'u23','u27','u30','u31','u35','u36','u44', 'u51','u52','u57','u59'

def convert_date(df, columns=['timestamp']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(
            'US/Eastern')
    return df

def read_data(name):
    conversation_data = pd.read_csv('../sensing/Conversations/conversation_'+name+'.csv')
    return conversation_data

uids_cleaned = []
for uid in uids:
    conv_data = read_data(uid)
    convert_date(conv_data, columns=['start_timestamp', 'end_timestamp'])
    conv_data['week_group'] = conv_data['start_timestamp'].dt.to_period('W-THU')
    conv_data.index = conv_data['start_timestamp']
    conv_data['timestamp'] = conv_data['start_timestamp']
    conv_data['date'] = pd.DatetimeIndex(conv_data['start_timestamp']).date
    group = conv_data.groupby(conv_data['date']).count()
    if len(group) <= 56:
        print(uid)
