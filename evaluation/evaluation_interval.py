import pandas as pd
import numpy as np
from datetime import datetime
import copy

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

#slots = 'p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p96'

slots = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96

coll_test_pattern = 'uid','context', 'date','zero'
coll_metrics = 'uid','context','date','week','zero'
test_pattern = np.zeros(97)

weeks = ['Week1','Week3', 'Week5']
contexts = ['SUNDAY_', 'MONDAY_','TUESDAY_', 'WEDNESDAY_', 'THURSDAY_', 'FRIDAY_', 'SATURDAY_']#, 'WEEK_', 'WEEK_END_']
#uids = 'u00', 'u02', 'u04', 'u05', 'u08', 'u09', 'u10', 'u12', 'u13', 'u14', 'u17', 'u23', 'u27', 'u30', 'u31', 'u36', 'u51', 'u53', 'u56', 'u57', 'u59'
uids = 'u00', 'u02', 'u05', 'u08', 'u09', 'u10','u12', 'u14', 'u17', 'u27', 'u30', 'u31',  'u51', 'u53',  'u57', 'u59', 'u35', 'u52'

#remover: , 'u04', 'u13''u23','u36','u56',
#adicionar:  u35, u52

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

#cria o dataframe os as leituras dos dias de teste
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

    #limpa os dados: remove os dias com menos de 12 instâncias
    conv_data['date_clean'] = pd.DatetimeIndex(conv_data['start_timestamp']).date
    group = conv_data.groupby(conv_data['date_clean']).count()
    group = group[group['start_timestamp'] < 12]

    # remove os dias com menos de 8 amostras
    '''print(conv_data)
    if(len(group) > 0):
        for idx in group.index:
            conv_data.drop([idx], inplace=True)
            print(idx)'''

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
    #print(test_dt)
    return test_dt

#print(read_pattern('pattern'))
#print(extractTestPattern('u00',0))

def get_parttern(uid,context,week, pattern):
    # ler os padrões
    patterns = read_pattern(pattern)
    # filtra por ('uid','week,'context)
    current_pattern = patterns[(patterns['uid'] == uid) & (patterns['week'] == week) & (patterns['context'] == context)]
    current_pattern.index = current_pattern['uid']
    current_pattern = current_pattern.loc[uid]
    current_pattern = current_pattern[(current_pattern == 0) | (current_pattern == 1)]
    return current_pattern

def define_parttern(current_pattern):
    intervals = []
    slots_test = []
    pattern_aux = 0
    interval = []
    for slot in current_pattern.index:
        # monta array de intervalos e slots
        if current_pattern[slot] == 1:
            interval.append((int(slot), 0))
            pattern_aux = 1
        if (current_pattern[slot] == 0) and (pattern_aux == 1):
            intervals.append(dict(interval))
            interval = []
            pattern_aux = 0
        if current_pattern[slot] == 0:
            slots_test.append((int(slot), 0))
            pattern_aux = 0
    dict_slots = dict(slots_test)
    return intervals, dict_slots

def get_values_metrics(uid,context,week, pattern):
    index = pd.MultiIndex.from_arrays(arrays=[[], [], []], names=['date', 'context', 'week'])
    coll_value_metrics = 'uid', 'acc_intervals', 'acc_slots', 'acc_interval_slot'
    value_metrics_dt = pd.DataFrame(columns=coll_value_metrics, index=index)

    current_pattern = get_parttern(uid,context,week,pattern)

    intervals, dict_slots = define_parttern(current_pattern)

    # carrega os dados de teste
    data_test = extractTestPattern(uid, context)
    data_test.index = data_test['date']

    # calcula a performace com os dados de teste
    for i in data_test.index:
        test = data_test.loc[i]
        date = test['date']
        test = test[test == 1]
        #cria a copia dos padrões
        intervals_aux =  copy.deepcopy(intervals)
        dict_slots_aux = copy.deepcopy(dict_slots)

        head = [uid]
        true_intervals = 0
        false_slot = 0
        for slot_test in test.index:
            #print(slot_test)
            for dict_interval in intervals_aux:
                if ((slot_test in dict_interval.keys()) & (0 in dict_interval.values())):
                    for key in dict_interval.keys():
                        dict_interval[key] = 1
                    #incrementa contador de intervalos acertados
                    true_intervals += 1
            if slot_test in dict_slots_aux:
                dict_slots_aux[slot_test] = 1
                #incrmenta o contador de slots errados
                false_slot += 1
        #print('intervalos acertados {}, slots errados {}'.format(true_intervals,false_slot))
        #print(intervals_aux)
        #print(dict_slots_aux)

        if(len(intervals) > 0):
            acc_interval = true_intervals/len(intervals)
            acc_slots = (len(dict_slots)-false_slot)/len(dict_slots)
            acc_interval_slot = (acc_interval+acc_slots)/2

            results = [acc_interval, acc_slots, acc_interval_slot]
            head.extend(results)
            value_metrics_dt.loc[(date,context,week)] = head

    return value_metrics_dt

#list = [(k,v) for k,v in dict_interval.items()]

#metrics_dt = get_values_metrics('u30','WEEK_','Week3','patterns')
#print(metrics_dt)



#values_dt = pd.DataFrame()
result_dt = pd.DataFrame()
'''for uid in uids:
    for ctx in contexts:
        for week in weeks:
            try:
                aux_dt = get_values_metrics(uid, ctx, week, 'patterns')
                if len(result_dt) <1:
                    result_dt = aux_dt
                else:
                    result_dt = result_dt.append(aux_dt)
            except:
                print('erro')
'''




#print(values_dt[['trueP','falseP','trueN', 'falseN']])
#print(values_dt[['acc', 'prec', 'recall', 'f1']])



















