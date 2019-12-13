import pandas as pd
import numpy as np
from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk",font_scale=3)


weeks = ['Week1','Week3', 'Week5']
contexts = ['SUNDAY_', 'MONDAY_','TUESDAY_', 'WEDNESDAY_', 'THURSDAY_', 'FRIDAY_', 'SATURDAY_']#, 'WEEK_', 'WEEK_END_']
metrics = ['acc_intervals','acc_slots','acc_interval_slot']
result_dt = pd.read_csv('../patterns/result_similarity.csv')#,index_col=[0,1,2])



def creat_chart_ctx(metric):
    fig = go.Figure()
    for week in weeks:
        group_ctx = result_dt.groupby(['context','week']).mean()
        group_aux = group_ctx[group_ctx.index.get_level_values('week') == week]
        fig.add_trace(go.Bar(
        x=group_aux.index.get_level_values('context'),
        y=group_aux[metric],
        name= week,
        ))
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text='Avaliação sobre contextos: {} - Métrica: {}'.format(week,metric))
    fig.show()

    # fig, ax = plt.subplots(figsize=(10, 9))
    '''s = sns.catplot(x='week', y=metric, kind="bar", data=group_ctx, height=25)
        s.savefig('../img/context/{}/{}_{}'.format(ctx, ctx, metric))
        plt.title('')
        plt.xlabel(ctx)
        plt.show()'''

creat_chart_ctx('lvt_similarity')

#creat_chart_weeks()
#for metric in metrics:
    #creat_chart_ctx(metric)