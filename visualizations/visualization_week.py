import pandas as pd
import numpy as np
from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

weeks = ['Week1','Week3', 'Week5']
contexts = ['SUNDAY_', 'MONDAY_','TUESDAY_', 'WEDNESDAY_', 'THURSDAY_', 'FRIDAY_', 'SATURDAY_']#, 'WEEK_', 'WEEK_END_']
metrics = ['acc_intervals','acc_slots','acc_interval_slot']
result_dt = pd.read_csv('../patterns/result_test_pattern.csv')#,index_col=[0,1,2])

def creat_chart_weeks():
    colors = 'lightslategray'
    for week in weeks:
        aux_dt = result_dt[result_dt['week'] == week]
        group_ctx = aux_dt.groupby(['context']).mean()
        print(group_ctx)
        group_ctx[week] = group_ctx.index
        print('-----------------------------------------')

        for metric in metrics:
            fig = go.Figure(data=[go.Bar(
                x=group_ctx.index,
                y=group_ctx[metric],
                marker_color=colors  # marker color can be a single color value or an iterable
            )])
            fig.update_layout(title_text='Avaliação da {} - Métrica: {}'.format(week,metric))
            fig.show()
            #fig, ax = plt.subplots(figsize=(10, 9))
            #s = sns.catplot(x=week, y=metric, kind="bar", data=group_ctx, height=25)
            #s.savefig('../img/{}/{}_{}'.format(week,week,metric))
            #plt.title('')
            #plt.show()

creat_chart_weeks()