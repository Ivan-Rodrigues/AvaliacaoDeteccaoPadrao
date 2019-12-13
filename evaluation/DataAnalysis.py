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
uids = 'u00','u02', 'u04','u05','u08',  'u08', 'u09','u10','u12','u13','u14','u16','u17','u19',\
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

array_aleatorio = range(1,100)

def convert_date(df, columns=['start_timestamp'], isCrossover=False):

    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True,).dt.tz_convert(
            'US/Eastern')
    if (isCrossover):
        df['start_timestamp'] = df['start_timestamp'].apply(lambda x: x + + timedelta(days=62))

    df['context'] = df['start_timestamp'].apply(lambda x: days[x.weekday()])
    df['slot'] = df['start_timestamp'].apply(lambda x: extractSlot(x,w))
    df['date'] = df['start_timestamp'].apply(lambda x: datetime.date(x))
    return df

def read_data(name):
    conversation_data = pd.read_csv('../sensing/Conversations/conversation_'+name+'.csv')
    return conversation_data


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
    sns.despine(offset=50, trim=True);
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
    no_socialization = 0
    return (count_intersect+no_socialization)/(count_union+no_socialization)

# calcula a similaridade entre leituras
def get_similarity_pattern_adapt(stream, uid):
    count = 0
    id =0
    index = []
    for i in stream.index:
        index.append(count)
        count += 1
    stream.index = index
    count = 0

    coll_similarity = 'uid', 'similarity'
    dt_similarity = pd.DataFrame(columns=coll_similarity)
    actual_pattern =[]
    for idx in stream.index:
        if id != len(stream) -1:
            if (idx == 0):
                actual_pattern = stream.loc[idx]
                pattern2 = stream.loc[idx+1]
                similarity = calc_jaccard(actual_pattern, pattern2)
                dt_similarity.loc[id + 1] = [uid, similarity]
                id += 1
                if calc_jaccard(actual_pattern, pattern2) < 0.6:
                    actual_pattern = pattern2
            else:
                pattern2 = stream.loc[idx+1]
                similarity = calc_jaccard(actual_pattern, pattern2)
                dt_similarity.loc[id+1] = [uid, similarity]
                id+=1
                if calc_jaccard(actual_pattern, pattern2) < 0.6:
                    actual_pattern = pattern2

    return dt_similarity


# calcula a similaridade entre leituras
def get_similarity_days(stream, uid):
    count = 0
    id =0
    index = []
    for i in stream.index:
        index.append(count)
        count += 1
    stream.index = index
    count = 0

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

def convert_index(stream):
    count = 0
    index = []
    for i in stream.index:
        index.append(count)
        count += 1
    stream.index = index
    return stream



# calcula a similaridade entre padrões e leituras
def get_similarity_pattern_days(stream, uid,  number_obs=3, is_adapt=False):
    id = 0
    stream = convert_index(stream)
    # primeiro padrão
    actual_pattern = stream.loc[number_obs]
    #apaga as observações do primeiro padrão
    stream = stream[stream.index > number_obs]
    aux_pattern=0
    coll_similarity = 'uid', 'similarity'
    dt_similarity = pd.DataFrame(columns=coll_similarity)
    for idx in stream.index:
        if (aux_pattern != number_obs ):
            actual_read = stream.loc[idx]
            similarity = calc_jaccard(actual_pattern, actual_read)
            dt_similarity.loc[id+1] = [uid, similarity]
            id+=1
            aux_pattern +=1
        else:
            if (calc_jaccard(actual_pattern, stream.loc[idx]) < 0.6) & is_adapt:
                actual_pattern = stream.loc[idx]
                print('change')
            aux_pattern =0
    return dt_similarity


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


#Calcula a similaridade utilizando uma janela deslizante (Padrão estável e outro reativo)
def get_similarity_slide_pattern(stream, num_obs, uid):
    stream = convert_index(stream)
    coll_similarity = 'uid', 'similarity'
    dt_similarity = pd.DataFrame(columns=coll_similarity)
    actual_pattern = []
    id=0
    for idx in stream.index:
        if  actual_pattern == []:
            sum_pattern =  [int(e1) + int(e2) + int(e3) for e1, e2, e3
               in zip_longest(stream.loc[0], stream.loc[1], stream.loc[2], fillvalue=0)]
            #padrão montado com 3 semanas
            actual_pattern = extract_patter(sum_pattern, [], num_obs=num_obs)

        else:
            if (idx+3 <= len(stream)-1):
                sum_pattern = [int(e1) + int(e2) + int(e3) for e1, e2, e3
                               in zip_longest(stream.loc[idx], stream.loc[idx+1], stream.loc[idx+2], fillvalue=0)]

                # padrão montado com 3 semanas
                pattern_test = extract_patter(sum_pattern, [], num_obs=num_obs)
                similarity = calc_jaccard(actual_pattern, pattern_test)
                #verifica se é necessário mudar o padrão
                if (similarity < 0.6):
                    actual_pattern = pattern_test
                    dt_similarity.loc[id] = [uid, similarity]
                    id += 1
                else:
                    dt_similarity.loc[id] = [uid, similarity]
                    id +=1
    return dt_similarity

def calc_statistic_similarity(sim_days, sim_patterns1,sim_patterns2,sim_patterns3,sim_patterns4,
                              sim_patterns_days1, sim_patterns_days2, sim_patterns_days3 , sim_patterns_days4,
                              sim_slide_pattern, sim_patterns_days_obs, sim_patterns_adapt):

    mean_days = sim_days['similarity'].mean()

    mean_patterns1 = sim_patterns1['similarity'].mean()
    mean_patterns2 = sim_patterns2['similarity'].mean()
    mean_patterns3 = sim_patterns3['similarity'].mean()
    mean_patterns4 = sim_patterns4['similarity'].mean()

    mean_patterns_day1 = sim_patterns_days1['similarity'].mean()
    mean_patterns_day2 = sim_patterns_days2['similarity'].mean()
    mean_patterns_day3 = sim_patterns_days3['similarity'].mean()
    mean_patterns_day4 = sim_patterns_days4['similarity'].mean()

    mean_slide_pattern = sim_slide_pattern['similarity'].mean()
    mean_patterns_days_obs = sim_patterns_days_obs['similarity'].mean()
    mean_patterns_adapt = sim_patterns_adapt['similarity'].mean()

    std_days = sim_days['similarity'].std()
    #std_patterns_day = sim_patterns_days['similarity'].std()
    std_slide_pattern = sim_slide_pattern['similarity'].std()
    std_patterns_days_obs = sim_patterns_days_obs['similarity'].std()
    std_patterns_adapt = sim_patterns_adapt['similarity'].std()

    md_days = sim_days['similarity'].median()
    #md_patterns_day = sim_patterns_days['similarity'].median()
    md_slide_pattern = sim_slide_pattern['similarity'].median()
    md_patterns_days_obs = sim_patterns_days_obs['similarity'].median()
    md_patterns_adapt = sim_patterns_adapt['similarity'].median()

    return [mean_days, mean_patterns1, mean_patterns2, mean_patterns3, mean_patterns4,
            mean_patterns_day1, mean_patterns_day2, mean_patterns_day3, mean_patterns_day4,
            mean_slide_pattern, mean_patterns_days_obs, mean_patterns_adapt,
            std_days, std_slide_pattern, std_patterns_days_obs, std_patterns_adapt,
            md_days,   md_slide_pattern, md_patterns_days_obs, md_patterns_adapt]

index = pd.MultiIndex.from_arrays(arrays=[[], []], names=['uid', 'context'])
coll_stast = 'uid','context','m_day', 'm_pattern1', 'm_pattern2', 'm_pattern3', 'm_pattern4', \
             'm_pattern_day1', 'm_pattern_day2','m_pattern_day3', 'm_pattern_day4', 'm_slide_pattern', 'm_pattern_day_obs', 'm_pattern_adapt', \
             'std_day',  'std_slide_pattern', 'std_pattern_day_obs','std_pattern_adapt', \
                'md_day', 'md_slide_pattern', 'md_pattern_day_obs', 'md_pattern_adapt'
result_dt = pd.DataFrame(columns=coll_stast,  index=index)

'''for usr in uids:
    print(usr)
    for context in days:
        #print(slide_pattern)

        #mix, só padrão, padrão considerando obs
        # fluxo de padrões com adaptação e dias considerando observáveis
        st_patterns_adapt = get_stream(usr, context, False, True, False, num_obs=3)
        #fluxo de padrões e dias considerando observáveis
        st_patterns_days_obs = get_stream(usr, context, True, False, True, 3)
        # fluxo de padrões e dias
        st_patterns_days1 = get_stream(usr, context, True, False, False, num_obs=1)
        st_patterns_days2 = get_stream(usr, context, True, False, False, num_obs=2)
        st_patterns_days3 = get_stream(usr, context, True, False, False, num_obs=3)
        st_patterns_days4 = get_stream(usr, context, True, False, False, num_obs=4)

        # fluxo de padrões
        st_patterns1 = get_stream(usr, context, False, True, False, num_obs=1)
        st_patterns2 = get_stream(usr, context, False, True, False, num_obs=2)
        st_patterns3 = get_stream(usr, context, False, True, False, num_obs=3)
        st_patterns4 = get_stream(usr, context, False, True, False, num_obs=4)

        # fluxo de padrões considerando observáveis
        st_patterns_obs = get_stream(usr, context, False, True, True, num_obs=3)
        # fluxo de dias
        st_days = get_stream(usr, context, False, False, False, num_obs=3)

        # recupera as similaridades entre padrões (dias e padrões são recuperados da mesma forma)
        sim_patterns_adapt = get_similarity_pattern_adapt(st_patterns3, usr)

        #recupera as similaridades entre padrões (dias e padrões são recuperados da mesma forma)
        sim_patterns1 = get_similarity_days(st_patterns1, usr)
        sim_patterns2 = get_similarity_days(st_patterns2, usr)
        sim_patterns3 = get_similarity_days(st_patterns3, usr)
        sim_patterns4 = get_similarity_days(st_patterns4, usr)

        #recupera a similaridade entre dias
        sim_days = get_similarity_days(st_days, usr)
        #recupera a similaridade entre padrões e dias
        sim_patterns_days1 = get_similarity_pattern_days(st_patterns_days1, usr, 1)
        sim_patterns_days2 = get_similarity_pattern_days(st_patterns_days2, usr, 2)
        sim_patterns_days3 = get_similarity_pattern_days(st_patterns_days3, usr, 3)
        sim_patterns_days4 = get_similarity_pattern_days(st_patterns_days4, usr, 4)

        #recupera a similaridade entre padrões considerando observáveis e dias
        sim_patterns_days_obs = get_similarity_pattern_days(st_patterns_days_obs, usr, 3)
        #recupera a similaridade entre padrões utilizando janela deslizante
        sim_slide_pattern = get_similarity_slide_pattern(st_days, 3, usr)

        #calcula média e desvio padrão das similaridades
        statistics = calc_statistic_similarity(sim_days, sim_patterns1,sim_patterns2,sim_patterns3,sim_patterns4,
                                               sim_patterns_days1, sim_patterns_days2, sim_patterns_days3 , sim_patterns_days4,
                                               sim_slide_pattern, sim_patterns_days_obs, sim_patterns_adapt)
        head = [usr, context]
        line_result = head.extend(statistics)
        #cria dataframe
        result_dt.loc[(usr,context), ['uid','context','m_day', 'm_pattern1', 'm_pattern2', 'm_pattern3', 'm_pattern4',
                                      'm_pattern_day1', 'm_pattern_day2','m_pattern_day3', 'm_pattern_day4', 'm_slide_pattern', 'm_pattern_day_obs', 'm_pattern_adapt',
             'std_day',  'std_slide_pattern', 'std_pattern_day_obs','std_pattern_adapt',
                'md_day', 'md_slide_pattern', 'md_pattern_day_obs', 'md_pattern_adapt']] = head


print(result_dt[['m_pattern_adapt']])
result_dt.to_csv('similarity.csv')'''


'''st_patterns_days_01 = get_stream('u57', days[1], True, False, False, 3)
chart_stream(st_patterns_days_01)
sim_pattern = get_similarity_pattern_days(st_patterns_days_01, 'u57', 3)
chart_similarity(sim_pattern,"test")



st_patterns_days_02 = get_stream('u03', days[1], False, False, False, 3)
chart_stream(st_patterns_days_02)
'''
#rm u03
st_patterns_days_01 = get_stream('u04', days[0], True, False, False, 3, False)
#chart_stream(st_patterns_days_01)
sim_patterns_days = get_similarity_pattern_days(st_patterns_days_01, 'u04',3, True)
print(sim_patterns_days)
chart_similarity(sim_patterns_days)

st_patterns_days_01 = get_stream('u04', days[0], True, False, False, 3, False)
#chart_stream(st_patterns_days_01)
sim_patterns_days = get_similarity_pattern_days(st_patterns_days_01, 'u04',3, False)
print(sim_patterns_days)
chart_similarity(sim_patterns_days)


st_patterns_days_01 = get_stream('u57', days[0], True, False, False, 2, False)
chart_stream(st_patterns_days_01)

st_patterns_days_01 = get_stream('u04', days[0], True, False, False, 2, True)
chart_stream(st_patterns_days_01)
sim_patterns_days = get_similarity_pattern_days(st_patterns_days_01, 'u04',2, True)
print(sim_patterns_days)
chart_similarity(sim_patterns_days)
#st_patterns_days_02 = get_stream('u09', days[0], True, False, False, 2)
#chart_stream(st_patterns_days_02)

#merge_pattern = st_patterns_days_01.append(st_patterns_days_02)
#chart_stream(merge_pattern)

#sim_patterns_days = get_similarity_days(merge_pattern, 'u00')
#print(sim_patterns_days)


#stream_patterns = get_stream('u04', 'MONDAY_', True, False, 3)
#similarity_patterns = get_similarity_days(stream_patterns, 'u04')
#chart_stream(stream_patterns)
#chart_similarity(similarity_patterns)

#print(result_dt[['m_day', 'm_pattern', 'm_pattern_day']])
#result_dt = result_dt.xs('SUNDAY_', level=1, axis=0, drop_level=False)
#print(result_dt[['m_day', 'm_pattern', 'm_pattern_day']])
#print(result_dt.stack().index[np.argmax(result_dt.values)])





