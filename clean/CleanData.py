import pandas as pd
from datetime import datetime

def convert_date(df, columns=['timestamp']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(
            'US/Eastern')
    return df

def read_data(name):
    conversation_data = pd.read_csv('../sensing/Conversations/conversation_'+name+'.csv')
    return conversation_data

user_id = ['u00','u01','u02','u03','u04','u05','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',
        'u23','u25','u24','u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45'
    ,'u46','u47','u49','u50','u51','u52','u53','u54','u56','u57','u58','u59']


uid_satisfy = []
uid_less_instance = []
uid_less_days = []
for uid in user_id:
    conv_data = read_data(uid)

    #remove usuários com menos de 792 amostras no total (12 em cada dia)
    if len(conv_data) >= 0:

        convert_date(conv_data, columns=['start_timestamp', 'end_timestamp'])

        #conv_data = conv_data[pd.DatetimeIndex(conv_data.index).date == datetime.strptime(str(group.index[4]),'%Y-%m-%d').date()]
        conv_data.index = pd.DatetimeIndex(conv_data['start_timestamp']).date
        group = conv_data.groupby(conv_data.index).count()
        if len(group) > 60:
            print('{}'.format(uid))
        #group = group[group['start_timestamp'] < 12]

        #remove os dias com menos de 8 amostras
        #for idx in group.index:
            #conv_data.drop([idx],inplace=True)

        # remove os usuários com menos de +/- 80% dos dias do estudo (52 dias)
        group_count = conv_data.groupby(conv_data.index).count()
        #print(group_count)
        '''if len(group_count) >= 52:
            print('{} satisfaz: {}'.format(uid,len(group_count)))
            uid_satisfy.append(uid)
        else:
            print('{} está fora: {}'.format(uid, len(group_count)))
            uid_less_days.append(uid)
    else:
        print('Poucas instâncias do {}: {}'.format(uid,len(conv_data)))
        uid_less_instance.append(uid)'''

#print('satisfaz {}:'.format(len(uid_satisfy)))
#print(uid_satisfy)

#print(conv_data[conv_data.index == datetime.datetime('2013-03-27')] )



#idx = conv_data.index.unique()
#group.index = idx






