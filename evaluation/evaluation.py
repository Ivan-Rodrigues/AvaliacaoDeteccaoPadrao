import pandas as pd
import numpy as np
from datetime import datetime

# extrair slot
def trunc(num):
   sp = str(num).split('.')
   return int(sp[0])

# extrai o número de slots referente aos minutos
def get_slot_minute(w, time):
    min = 60 / (1 / w)
    if time < min:
        return 1
    elif time % min == 0:
        return time / min
    else:
        return trunc(time / min) + 1

# Extrai o slot referente ao timestamp do evento
def extractSlot(startTime, w):
    startTime = datetime.time(startTime)

    hourEvent = startTime.hour
    minuteEvent = startTime.minute
    slot = (hourEvent / w) + get_slot_minute(w, minuteEvent)
    slot = int(slot)
    return slot

slots = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96
coll_test_pattern = 'uid','context', 'date','zero'
coll_metrics = 'uid','context','date','week','zero'
test_pattern = np.zeros(97)

weeks = ['Week1','Week3', 'Week5']
uids = ['u00','u02','u05']

days = [
    'MONDAY_',
    'TUESDAY_',
    'WEDNESDAY_',
    'THURSDAY_',
    'FRIDAY_',
    'SATURDAY_',
    'SUNDAY_'
]


def convert_date(df, columns=['start_timestamp']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True,).dt.tz_convert(
            'US/Eastern')
    df['context'] = df['start_timestamp'].apply(lambda x: days[x.weekday()])
    return df

def read_data(name):
    conversation_data = pd.read_csv('../sensing/Conversations/conversation_'+name+'.csv')
    return conversation_data
#padrões

def read_pattern(name):
    return pd.read_csv('../patterns/{}.csv'.format(name))


#cria o dataframe os as leitus dos dias de teste
def extractTestPattern(uid, context):
    #context = days[dayOfWeek]

    #dados de teste
    conv_data = read_data(uid)
    convert_date(conv_data, columns=['start_timestamp', 'end_timestamp'])
    conv_data.index = conv_data['start_timestamp']
    conv_data.index.name = 'idx'
    week_teste = (conv_data['start_timestamp'] >= '2013-05-01') & (conv_data['start_timestamp'] < '2013-05-28')
    conv_data = conv_data.loc[week_teste]

    #filtra por contexto
    conv_data = conv_data[conv_data['context'] == context]

    conv_data['date'] = conv_data['start_timestamp'].apply(lambda x : datetime.date(x))

    #cria o dataframe de teste
    test_dt = pd.DataFrame(columns=coll_test_pattern+slots)
    conv_data = conv_data.reset_index()

    for i in conv_data['date'].unique():
        data_aux = conv_data[conv_data['date'] == i]
        test = np.zeros(97)
        #print(test)

        for j in data_aux.index:
            slot = extractSlot(data_aux.loc[j]['start_timestamp'],0.25)
            test[slot] = 1

        #date = data_aux.loc[str(i)]['date']
        head = [uid, context, str(i)]
        head.extend(test)
        test_dt.loc[i] = head
    return test_dt

#print(read_pattern('pattern_monday'))
#print(extractTestPattern('u00',0))

def get_values_metrics(uid,context,week, pattern):
    index = pd.MultiIndex.from_arrays(arrays=[[], [], []], names=['date', 'context', 'week'])
    coll_value_metrics = 'uid', 'trueP', 'falseP', 'trueN', 'falseN', 'acc', 'prec', 'recall', 'f1'

    value_metrics_dt = pd.DataFrame(columns=coll_value_metrics, index=index)

    # ler os padrões
    patterns = read_pattern(pattern)

    # filtra por ('uid','week,'context)
    current_pattern = patterns[(patterns['uid'] == uid) & (patterns['week'] == week) & (patterns['context'] == context)]
    current_pattern.index = current_pattern['uid']

    #carrega os dados de teste
    data_test = extractTestPattern(uid, context)
    data_test.index = data_test['date']


    #calcula a performace a cada dia
    for i in data_test.index:
        test = data_test.loc[i]
        date = test['date']
        ptt = current_pattern.loc[uid]

        head = [uid]
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for slot in slots:
            # não ocorreu evento (0) -> 1 = tn e 0 fp
            if test[slot] == 0:
                if ptt[slot] == 0:
                    tn +=1
                else:
                    fp +=1
            # ocorreu evento (1) -> 1 = tp e 0 fn
            else:
                if ptt[slot] == 1:
                    tp +=1
                else:
                    fn +=1

        correct = tp + tn
        all = correct + fp + fn
        acuracia = correct/all
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        try:
            f1_score = (2*precision*recall)/(precision+recall)
        except:
            f1_score=0
        results = [tp, fp,tn,fn,acuracia,precision,recall,f1_score]
        #print(results)
        head.extend(results)
        value_metrics_dt.loc[(date,context,week)] = head

    return value_metrics_dt

#print(get_values_metrics('u05','MONDAY_','Week3','pattern_monday')[['trueP', 'falseP', 'trueN', 'falseN', 'acc', 'prec', 'recall', 'f1']])


#print(get_values_metrics('u05','MONDAY_','Week5','pattern_monday')[['trueP', 'falseP', 'trueN', 'falseN', 'acc', 'prec', 'recall', 'f1']])

#print(values_dt[values_dt['uid'] == 'u00'][['trueP','trueN','falseP', 'falseN','acc', 'prec', 'recall']])


def printStatistics(uid,week, values_dt):

    print('\n Média dos desempenhos')
    group_mean = values_dt.groupby(['uid','week']).mean()
    print(group_mean)

    values_dt = values_dt[values_dt['uid'] == uid]
    values_dt = values_dt[values_dt.index.get_level_values('week').isin([week])]
    #values_dt = values_dt[values_dt.index.get_level_values('date').isin(['2013-05-27'])]

    print('\n Resumo dos acertos')
    print(values_dt[['uid','trueP','falseP','trueN', 'falseN']])

    print('\n Resumo dos desempenhos')
    print(values_dt[['uid','acc', 'prec', 'recall','f1']])

    print('\n Valores máximos:')
    print('{} - {}: max acc: {}, max prec: {}, max recall: {}, max f1: {}'.format(uid, week, values_dt['acc'].max(),
                                                                              values_dt['prec'].max(),
                                                                              values_dt['recall'].max(),
                                                                              values_dt['f1'].max()))

    print('\n Valores mínimos:')
    print('{} - {}: min acc: {}, min prec: {}, min recall: {}, min f1: {}'.format(uid, week, values_dt['acc'].min(),
                                                                              values_dt['prec'].min(),
                                                                              values_dt['recall'].min(),
                                                                              values_dt['f1'].min()))





values_dt = pd.DataFrame()

for u in uids:
    for i in weeks:
        uid = u
        week = i
        pattern = 'pattern_monday'
        context = days[0]
        values_dt = values_dt.append(get_values_metrics(uid, context, week,'pattern_monday'))
        #print(get_values_metrics(uid, context, week,'pattern_monday')['prec'])


uid = 'u02'
week = 'Week3'

printStatistics(uid,week,values_dt)


#print(values_dt[['trueP','falseP','trueN', 'falseN']])
#print(values_dt[['acc', 'prec', 'recall', 'f1']])



















