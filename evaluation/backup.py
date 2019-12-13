import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
slots = 'p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p60','p61','p62','p63','p64','p65','p66','p67','p68','p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81','p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95','p96'
coll_test_pattern = 'uid','context', 'date','zero'
coll_metrics = 'uid','context','date','week','zero'
test_pattern = np.zeros(97)


def calc_performace(uid, context, week, data_test, current_pattern):

    perform_dt = pd.DataFrame(columns=coll_metrics + slots)
    result = []
    for i in data_test.index:
        data = data_test.loc[i]
        for j in slots:
            if data[j] == current_pattern.loc[uid][j]:
                result.append(1)
            else:
                result.append(0)
        date = data['date']
        head = [uid, context, str(date), week, context]
        head.extend(result)
        perform_dt.loc['{} - {}'.format(date, week)] = head
        result=[]
    return perform_dt


weeks = ['Week1','Week3', 'Week5']
contexts = ['SUNDAY_', 'MONDAY_','TUESDAY_', 'WEDNESDAY_', 'THURSDAY_', 'FRIDAY_', 'SATURDAY_']#, 'WEEK_', 'WEEK_END_']

metrics = ['acc_intervals','acc_slots','acc_interval_slot']
result_dt = pd.read_csv('../patterns/result_test_pattern.csv')#,index_col=[0,1,2])

def creat_chart_weeks():
    for week in weeks:
        aux_dt = result_dt[result_dt['week'] == week]
        group_ctx = aux_dt.groupby(['context']).mean()
        print(group_ctx)
        group_ctx[week] = group_ctx.index
        print('-----------------------------------------')
        group_ctx.index.name = '\n {}'.format(week)
        for metric in metrics:
            #fig, ax = plt.subplots(figsize=(10, 9))
            s = sns.catplot(x=week, y=metric, kind="bar", data=group_ctx, height=25)
            s.savefig('../img/{}/{}_{}'.format(week,week,metric))
            plt.title('')
            plt.show()

def creat_chart_ctx():
    for ctx in contexts:
        aux_dt = result_dt[result_dt['context'] == ctx]
        group_ctx = aux_dt.groupby(['week','context']).mean()
        group_ctx['week'] = ['Week1','Week3','Week5']
        group_ctx.index = ['Week1','Week3','Week5']
        print(group_ctx)

        #group_ctx.index.name = '\n {}'.format()
        for metric in metrics:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=group_ctx['context'],
                y=[14, 25, 16, 18, 22, 19, 15],
                name='Week 1',
                marker_color='indianred'
            ))

            fig.add_trace(go.Bar(
                x=group_ctx['context'],
                y=[19, 15, 14, 10, 12, 12, 16],
                name='Week 3',
                marker_color='lightsalmon'
            ))

            # Here we modify the tickangle of the xaxis, resulting in rotated labels.
            fig.update_layout(barmode='group', xaxis_tickangle=-45)
            fig.show()

            # fig, ax = plt.subplots(figsize=(10, 9))
            '''s = sns.catplot(x='week', y=metric, kind="bar", data=group_ctx, height=25)
            s.savefig('../img/context/{}/{}_{}'.format(ctx, ctx, metric))
            plt.title('')
            plt.xlabel(ctx)
            plt.show()'''
