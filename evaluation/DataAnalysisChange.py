from datetime import datetime
from random import randint
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from scipy import signal
from itertools import zip_longest
from datetime import timedelta

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 28}
plt.rc('font', **font)

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

slots = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48#,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96
#uids = 'u01','u15','u16', 'u17'#, 'u08', 'u09', 'u10', 'u12'#, 'u13', 'u14', 'u17', 'u23', 'u27', 'u30', 'u31', 'u36', 'u51', 'u53', 'u56', 'u57', 'u59'

#uids = 'u00','u01','u02', 'u03', 'u04','u05','u08', 'u07', 'u08', 'u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',\
 #      'u23','u27','u30','u31','u32','u33','u34','u35','u36', 'u41','u42','u43','u44','u45','u46', 'u47', 'u49', 'u50', 'u51','u52','u54','u57','u58','u59'
uids = 'u00','u02', 'u04','u05','u08', 'u09','u10','u12','u13','u14','u16','u17','u19',\
       'u23','u27','u30','u31','u35','u36','u44', 'u51','u52','u57','u59'
uid = ['u04']

days = [
    'MONDAY_',
    'TUESDAY_',
    'WEDNESDAY_',
    'THURSDAY_',
    'FRIDAY_',
    'SATURDAY_',
    'SUNDAY_'
]

w=0.5
def chart_similarity(similaty_dt, title='Observações'):
    # Draw plot
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    ax.vlines(x=similaty_dt.index, ymin=0, ymax=similaty_dt.similarity, color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=similaty_dt.index, y=similaty_dt.similarity, s=75, color='firebrick', alpha=0.7)

    # Title, Label, Ticks and Ylim
    ax.set_title('Similaridade entre {}'.format(title), fontdict={'size': 22})
    ax.set_ylabel('Índice de Similaridade')
    ax.set_xticks(similaty_dt.index)
    ax.set_xticklabels(similaty_dt.index, rotation=60, fontdict={'horizontalalignment': 'right', 'size': 12})
    ax.set_ylim(0, 1)

    # Annotate
    for row in similaty_dt.itertuples():
        ax.text(row.Index, row.similarity + .05, s=round(row.similarity, 2), horizontalalignment='center', verticalalignment='bottom',
                fontsize=14)
    sns.despine(offset=50, trim=True)
    plt.show()

def chart_stream(stream_dt):
    # heatmap
    plt.figure(figsize=(50, 20), dpi=80)
    sns.heatmap(stream_dt, xticklabels=stream_dt.columns, cbar=False,
                cmap='RdYlGn', center=0, annot=True)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.table = 100

    sns.despine(left=True, bottom=True)

    plt.show()

def convert_date(df, columns=['start_timestamp'], isCrossover=False):

    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True,).dt.tz_convert(
            'US/Eastern')
    if (isCrossover):
        df['start_timestamp'] = df['start_timestamp'].apply(lambda x: x + timedelta(days=62))

    df['context'] = df['start_timestamp'].apply(lambda x: days[x.weekday()])
    df['slot'] = df['start_timestamp'].apply(lambda x: extractSlot(x,w))
    df['date'] = df['start_timestamp'].apply(lambda x: datetime.date(x))
    return df

def read_data(name):
    conversation_data = pd.read_csv('../sensing/Conversations/conversation_'+name+'.csv')
    return conversation_data

#Extrai o padrão de sociabilidade
def extract_patter(stream, sum_obs=[], num_obs=3):
    if sum_obs != []:
        sum_obs = [int(e1) + int(e2) + int(e3)+ int(e4) + int(e5)  for e1, e2, e3, e4, e5
                   in zip_longest(sum_obs[0], sum_obs[1], sum_obs[2], sum_obs[3], sum_obs[4], fillvalue=0)]
    sum_stream = sum(stream)
    support = 0.02
    phi = 0.7
    interval = []
    slots = []
    #O evento deve aparecer em pelo menos 60% dos dias
    th_obs = (num_obs/100) * 50
    th_candidate = sum_stream * phi * 1/ (24/w)
    th_interval = sum_stream * support
    pattern_test = np.zeros(49)
    pattern = np.zeros(49)

    #print("Candidate Thshould: {}".format(th_candidate))
    #print("Interval Thshould {}".format(th_interval))

    for i in range(0,49):
        #considera apenas a contagem de eventos
        if (sum_obs == []):
            if stream[i] >= th_candidate:
                pattern_test[i] = stream[i]
        else:
            if (stream[i] >= th_candidate) and (sum_obs[i] >= th_obs):
                pattern_test[i] = stream[i]

    for i in range(0, 49):
        if (pattern_test[i] != 0):
            interval.append(pattern_test[i])
            slots.append(i)
        else:
            if len(interval) > 0:
                if sum(interval) >= th_interval:
                    for s in slots:
                        pattern[s] = stream[s]
            interval =[]
            slots = []
    if len(interval) > 0:
        if sum(interval) >= th_interval:
            for s in slots:
                pattern[s] = stream[s]

    #print(pattern)
    return pattern


#compara os padrões utilizando a similaridade de jaccard
def calc_jaccard(pattern1, pattern2):
    count_intersect = 0
    count_union = 0
    for i in range(1, 49):
        if ((pattern1[i] != 0) & (pattern2[i] != 0)):
            count_intersect += 1
            count_union += 1
        else:
            if (((pattern1[i] == 0) & (pattern2[i] != 0)) | ((pattern1[i] != 0) & (pattern2[i] == 0))):
                count_union += 1

    return (count_intersect) / (count_union)
    #return (count_intersect+48-count_union)/(count_union+48-count_union)



# cria o dataframe das leituras e dos padrões
def get_stream(uid, context, is_mix=False, is_pattern=False, is_pattern_obs=False, num_obs=3, isCrossOver=False):
    # dados de teste
    conv_data = read_data(uid)
    convert_date(conv_data, columns=['start_timestamp', 'end_timestamp'])
    conv_data.index = conv_data['start_timestamp']
    conv_data.index.name = 'idx'

    if (isCrossOver):
        conv_extend = read_data('u57')
        convert_date(conv_extend, columns=['start_timestamp', 'end_timestamp'])
        conv_extend['start_timestamp'] = conv_extend['start_timestamp'] + timedelta(days=100)
        conv_extend.index = conv_extend['start_timestamp']
        conv_extend['date'] = conv_extend['start_timestamp'].apply(lambda x: datetime.date(x))
        conv_extend.index.name = 'idx'
        conv_data = conv_data.append(conv_extend)



    stream_day_dt = pd.DataFrame(columns=slots)
    stream_patterns_dt = pd.DataFrame(columns=slots)
    conv_data = conv_data.reset_index()

    # define o contexto
    conv_data = conv_data[conv_data['context'] == context]

    # contagem de eventos por slot para detectar o padrão
    sum_slots = np.zeros(49)

    #contagem de observáveis por dia (Até 5 observáveis)
    sum_observaveis = [np.zeros(49),np.zeros(49),np.zeros(49),np.zeros(49),np.zeros(49)]
    id =0
    count_day=0
    for i in conv_data['date'].unique():
        data_aux = conv_data[conv_data['date'] == i]
        test = np.zeros(49)
        count_day += 1
        for j in data_aux.index:
            #Conta os eventos do dia
            slot = data_aux.loc[j]['slot']
            test[slot] += 1

            # Conta os eventos em 3 observações
            sum_slots[slot] = sum_slots[slot]+1
            #atualiza a contagem de uma observação
            sum_observaveis[count_day][slot] = 1

        stream_day_dt.loc[i] = test
        if count_day == num_obs:
            if(is_mix | is_pattern):
                if (is_pattern_obs):
                    pattern = extract_patter(sum_slots, sum_observaveis, num_obs=num_obs)
                else:
                    pattern = extract_patter(sum_slots,[], num_obs)
                idx_pattern = randint(0, 1000)
                id +=1
                stream_day_dt.loc['Pattern:{}'.format(id)] = pattern
                stream_patterns_dt.loc['Pattern:{}'.format(id)] = pattern
                sum_slots = np.zeros(49)
            count_day = 0

    if is_pattern == False:
        return stream_day_dt
    else:
        return stream_patterns_dt

def convert_index(stream):
    count = 0
    index = []
    for i in stream.index:
        index.append(count)
        count += 1
    stream.index = index
    return stream

# calcula a similaridade entre leituras
def get_similarity_days(stream, uid):
    stream = convert_index(stream)
    id=0
    coll_similarity = 'uid', 'similarity'
    dt_similarity = pd.DataFrame(columns=coll_similarity)
    for idx in stream.index:
        if (idx != 0):
            pattern1 = stream.loc[idx-1]
            pattern2 = stream.loc[idx]
            similarity = calc_jaccard(pattern1, pattern2)
            dt_similarity.loc[id+1] = [uid, similarity]
            id+=1
    return dt_similarity



#Calcula a similaridade utilizando uma janela deslizante (Padrão estável e outro reativo)
def get_similarity_slide_pattern(stream, num_obs, uid):
    stream = convert_index(stream)
    coll_similarity = 'uid', 'similarity', 'change'
    dt_similarity = pd.DataFrame(columns=coll_similarity)
    actual_pattern = []
    id=0
    for idx in stream.index:
        if  actual_pattern == []:
            if num_obs == 2:
                sum_pattern = [int(e1) + int(e2) for e1, e2
                               in zip_longest(stream.loc[0], stream.loc[1], fillvalue=0)]
            else:
                sum_pattern =  [int(e1) + int(e2) + int(e3) for e1, e2, e3
                    in zip_longest(stream.loc[0], stream.loc[1], stream.loc[2], fillvalue=0)]
            #padrão montado com 3 semanas
            actual_pattern = extract_patter(sum_pattern, [], num_obs=num_obs)

        else:
            if num_obs == 2:
                pattern_size = 2
            else:
                pattern_size = 3
            if (idx+pattern_size <= len(stream)-1):
                if num_obs == 2:
                    sum_pattern = [int(e1) + int(e2) for e1, e2
                                   in zip_longest(stream.loc[idx], stream.loc[idx + 1], fillvalue=0)]
                else:
                    sum_pattern = [int(e1) + int(e2) + int(e3) for e1, e2, e3
                               in zip_longest(stream.loc[idx], stream.loc[idx+1], stream.loc[idx+2], fillvalue=0)]

                # padrão montado
                pattern_test = extract_patter(sum_pattern, [], num_obs=num_obs)
                similarity = calc_jaccard(actual_pattern, pattern_test)
                #verifica se é necessário mudar o padrão
                if (similarity < 0.55):
                    actual_pattern = pattern_test
                    dt_similarity.loc[id] = [uid, similarity, 'True']
                    id += 1
                else:
                    dt_similarity.loc[id] = [uid, similarity, 'False']
                    id +=1
    return dt_similarity


# calcula a similaridade entre padrões e leituras
def get_similarity_pattern_days_adapt(stream, uid,  number_obs=3, is_adapt=False):
    id = 0
    date = stream.index
    stream = convert_index(stream)
    # primeiro padrão
    actual_pattern = stream.loc[number_obs]
    #apaga as observações do primeiro padrão
    stream = stream[stream.index > number_obs]
    aux_pattern=0
    coll_similarity = 'date','uid', 'similarity', 'change'
    dt_similarity = pd.DataFrame(columns=coll_similarity)
    for idx in stream.index:
        if (aux_pattern != number_obs ):
            actual_read = stream.loc[idx]
            similarity = calc_jaccard(actual_pattern, actual_read)
            if similarity < 0.5:
                change = 'Comportamento anômalo'
            else:
                change = 'Comportamento normal'
            dt_similarity.loc[id+1] = [date[idx], uid, similarity, 'Leitura: {}'.format(change)]
            id+=1
            aux_pattern +=1
        else:
            similarity = calc_jaccard(actual_pattern, stream.loc[idx])
            if(is_adapt):
                if (similarity < 0.6):
                    actual_pattern = stream.loc[idx]
                    dt_similarity.loc[id + 1] = [date[idx], uid, similarity, 'Padrão: Mudou']
                    id += 1
                else:
                    dt_similarity.loc[id + 1] = [date[idx], uid, similarity, 'Padrão: Manteve']
                    id += 1
            else:
                actual_pattern = stream.loc[idx]

            aux_pattern =0

    return dt_similarity

#Padrões estáveis e interpretáveis (u09 - days[0]) (27 - [6] 0.5 e 0.6, phi=1)

#Configurações do experimento
usr = 'u27'
ctx = days[6]

#imprime padrão com 2 semanas
st_pattern = get_stream(usr, ctx, True, False, True,  num_obs=2)
chart_stream(st_pattern)

#imprime padrão com 3 semanas
st_pattern = get_stream(usr, ctx, True, False, True,  num_obs=3)
chart_stream(st_pattern)

print('Similaridade entre dias')
st_pattern = get_stream(usr, ctx, False, False, False)
chart_stream(st_pattern)
sim_slide_pattern = get_similarity_days(st_pattern,usr)
print(sim_slide_pattern)

#chart_similarity(sim_slide_pattern)

print('\nDetectando change com slide pattern de 2 semanas')
st_pattern = get_stream(usr, ctx, False, False, True, 2)
sim_slide_pattern = get_similarity_slide_pattern(stream=st_pattern,num_obs=2, uid=usr)
print(sim_slide_pattern)

print('\nDetectando change com slide pattern de 3 semanas')
st_pattern = get_stream(usr, ctx, False, False, is_pattern_obs=False, num_obs=3)
sim_slide_pattern = get_similarity_slide_pattern(stream=st_pattern,num_obs=3, uid=usr)
print(sim_slide_pattern)

print('\nsimilaridade entre padrões usando 2 semanas com adaptação')
st_pattern = get_stream(uid=usr, context=ctx, is_mix=True, is_pattern=False, is_pattern_obs=True, num_obs=2)
sim_slide_pattern = get_similarity_pattern_days_adapt(st_pattern,usr, number_obs=2, is_adapt=True)
print(sim_slide_pattern)

print('\nsimilaridade entre padrões usando 3 semanas com adaptação')
st_pattern = get_stream(uid=usr, context=ctx, is_mix=True, is_pattern=False, is_pattern_obs=True, num_obs=3)
sim_slide_pattern = get_similarity_pattern_days_adapt(st_pattern,usr, number_obs=3, is_adapt=True)
print(sim_slide_pattern)


print('\nsimilaridade entre padrões usando 2 semanas com adaptação')
st_pattern = get_stream(uid=usr, context=ctx, is_mix=True, is_pattern=False, is_pattern_obs=True, num_obs=2)
st_pattern_test = get_stream(uid='u04', context=days[2], is_mix=True, is_pattern=False, is_pattern_obs=True, num_obs=2)
merge = st_pattern.append(st_pattern_test)
sim_slide_pattern = get_similarity_pattern_days_adapt(merge,usr, number_obs=2, is_adapt=True)
print(sim_slide_pattern)







#chart_similarity(sim_slide_pattern)

#print('similaridade entre padrões usando janela deslizante de 2 semanas (obs)')
#print(sim_slide_pattern)

#st_pattern = get_stream('u09', days[0], True, False, False, num_obs=2)
#sim_slide_pattern = get_similarity_pattern_days_adapt(st_pattern,'u09', number_obs=2, is_adapt=True)
#print('similaridade entre padrões usando janela deslizante de 2 semanas (obs)')
#print(sim_slide_pattern)



'''st_days = get_stream('u00', days[1], True, False, False, num_obs=3, isCrossOver=False)
chart_stream(st_days)
st_days = get_stream('u00', days[1], False, True, False, num_obs=3, isCrossOver=False)
chart_stream(st_days)
print(get_similarity_days(st_days,'u00'))
st_days_2 = get_stream('u00', days[6], False, True, False, num_obs=3, isCrossOver=False)
st_days_2.index = st_days_2.index+ timedelta(days=100)
chart_stream(st_days_2)'''
#(u04 -> u59)
#(u59 -> u00)

'''merge = st_days.append(st_days_2)
merge ['date'] = merge.index
lista = list(range(len(merge)))
merge.index = lista

merge_dt = get_similarity_slide_pattern(merge,3,'u04')
merge_dt = merge_dt[['similarity', 'date', 'change']]

merge_dt.index = merge_dt['date']
merge_dt = merge_dt['similarity']
plt.figure(figsize=(20, 5), dpi=100)
sns.lineplot(data=merge_dt, palette="tab10", linewidth=2.5, style="choice", markers=True)
plt.ylim([0,1])'''
#plt.legend(bbox_to_anchor=(1.08, 1), loc=1, borderaxespad=0.)
#plt.title('Context: {}')
plt.show()