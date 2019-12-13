import paho.mqtt.client as mqtt
import pandas as pd
from datetime import timedelta

#broker = 'iot.eclipse.org'
broker = '127.0.0.1'
pub_topic = 'social'
sub_topic_abnormal = 'com/lsdi/sociability/Abnormal'
sub_topic_change = 'com/lsdi/sociability/Change'
sub_topic_pattern = 'com/lsdi/sociability/Pattern'

def convert_date(df, columns=['timestamp']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(
            'US/Eastern')
    return df

def read_data(name):
    conversation_data = pd.read_csv('../sensing/Conversations/conversation_'+name+'.csv')
    return conversation_data

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    #Subscreve
    client.subscribe(sub_topic_abnormal)
    client.subscribe(sub_topic_change)
    client.subscribe(sub_topic_pattern)
    print("Publishing message to topic", pub_topic)


def on_message(client, userdata, message):
    socialEvent = message.payload.decode("utf-8")
    print("\n message received ", socialEvent)


client = mqtt.Client()

#broker_address="iot.eclipse.org"
client = mqtt.Client(client_id="Ivan", clean_session=True, userdata=None, transport="tcp")
print("connecting to broker")
client.connect(broker,port=1883,) #connect to broker

client.on_connect = on_connect
client.on_message = on_message
#remover 32
uids = 'u00', 'u02', 'u04', 'u05', 'u08', 'u09', 'u10', 'u12', 'u13', 'u14', 'u17', 'u23', 'u27', 'u30', 'u31', 'u36', 'u51', 'u53', 'u56', 'u57', 'u59'

#possuem padrões de 3 semanas similares
#u04, u12
#Thusday, Monday, weekend
#Padrão diferente (u00 -> u59) (u04 -> u59)
uid = 'u04'

conv_data = read_data(uid)
convert_date(conv_data, columns=['start_timestamp', 'end_timestamp'])
conv_data['week_group'] = conv_data['start_timestamp'].dt.to_period('W-THU')
conv_data.index = conv_data['start_timestamp']
#print(conv_data['week_group'].unique())

week1 = (conv_data['start_timestamp'] >= '2013-04-24') & (conv_data['start_timestamp'] < '2013-05-02')
week3 = (conv_data['start_timestamp'] >= '2013-04-10') & (conv_data['start_timestamp'] < '2013-05-02')

week6 = (conv_data['start_timestamp'] >= '2013-03-27') & (conv_data['start_timestamp'] < '2013-05-09')

#atribui os dados da semana
#conv_data = conv_data.loc[week6]

#limpa os dados: remove os dias com menos de 12 instâncias

conv_data['timestamp']= conv_data['start_timestamp']
conv_data['date'] = pd.DatetimeIndex(conv_data['start_timestamp']).date
group = conv_data.groupby(conv_data['date']).count()
#group = group[group['start_timestamp'] < 12]
conv_data.index = conv_data['date']

arq = open('../pattern_{}.txt'.format(uid), 'w')
events = []

#client.loop_forever()
i = 0
for timestamp in conv_data['start_timestamp']:
    #time = conv_data.loc[row]['start_timestamp']
    #prox_time = conv_data.loc[row + 1]['start_timestamp']
    #print(time)
    #time = str((timestamp + timedelta(days=42)))[0:19]
    #time = str((timestamp + timedelta(days=85)))[0:19]
    time = str((timestamp))[0:19]
    #print(time)
    #end_time = conv_data.loc[row]['end_timestamp']
    client.publish(pub_topic, payload=time, qos=2, retain=True)
    events.append('{}{}'.format(time,'\n'))
    i=i+1


arq.writelines(events)
arq.close()

client.loop_forever()