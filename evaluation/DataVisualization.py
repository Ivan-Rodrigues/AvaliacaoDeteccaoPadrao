from datetime import datetime
from random import randint
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from scipy import signal

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)

slots = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48#,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96
#uids = 'u01','u15','u16', 'u17'#, 'u08', 'u09', 'u10', 'u12'#, 'u13', 'u14', 'u17', 'u23', 'u27', 'u30', 'u31', 'u36', 'u51', 'u53', 'u56', 'u57', 'u59'

#uids = 'u00','u01','u02', 'u03', 'u04','u05','u08', 'u07', 'u08', 'u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20',\
      # 'u23','u27','u30','u31','u32','u33','u34','u35','u36', 'u41','u42','u43','u44','u45','u46', 'u47', 'u49', 'u50', 'u51','u52','u54','u57','u58','u59'

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

    # Decorations
    plt.title('Similaridade entre Padrões e Dias Posteriores', fontsize=22)
    plt.show()

def chart_stream(stream_dt):
    # heatmap
    plt.figure(figsize=(50, 20), dpi=80)
    sns.heatmap(stream_dt, xticklabels=stream_dt.columns, yticklabels=False,
                cmap='RdYlGn', center=0, annot=True, linewidths=.5)

    plt.title('Data Stream', fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.table = 100
    plt.show()

result_dt = pd.read_csv("similarity.csv", index_col=[0,1], skipinitialspace=True)
#print(result_dt)
#print(result_dt[['m_day', 'm_pattern', 'm_pattern_day']])
#result_dt = result_dt.xs(days[0], level=1, axis=0, drop_level=False)

def chart_mult_var(result_dt, var='m_pattern_day2'):
    sim_context_dt = pd.DataFrame(index=uids, columns=['MONDAY_'])

    for ctx in days:
        context_dt = result_dt.xs(ctx, level=1, axis=0, drop_level=True)[var]
        sim_context_dt[ctx] = context_dt

    sim_context_dt['uid'] =  uids

    # Load the dataset
  # Make the PairGrid
    g = sns.PairGrid(sim_context_dt.sort_values("MONDAY_", ascending=False),
                     x_vars=sim_context_dt.columns[:7], y_vars=["uid"],
                     height=10, aspect=.25)

    # Draw a dot plot using the stripplot function
    g.map(sns.stripplot, size=10, orient="h",
          palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

    # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(0, 1), xlabel='Similarity \n ({})'.format(var), ylabel="")

    # Use semantically meaningful titles for the columns
    titles = ["MONDAY_", "TUESDAY_", "WEDNESDEY_",
              "THURSDAY_", "FRIDAY_", "SATURDAY_", "SUNDAY_"]

    for ax, title in zip(g.axes.flat, titles):
        # Set a different title for each axes
        ax.set(title=title)
        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    plt.show()



def chart_pattern_rot(result_dt, var):
    # Cria gráficos para demonstrar a relação entre padrões e rotina
    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)
    print(result_dt)
    sns.scatterplot(x='m_day', y=var, data=result_dt, palette='tab10', hue='context', ax=ax,
                    style="context")
    # Decorations
    plt.title('Relação entre rotina e padrões', fontsize=22)

    kws = dict(s=50, linewidth=2.5, edgecolor="w")
    g = sns.FacetGrid(result_dt, col="context", palette="Set1", col_wrap=3, aspect=1.5)
    g = (g.map(sns.scatterplot, "m_day", var, **kws).add_legend())

    plt.show()

def chart_ptn_day_high(result_dt, var):
    #conta usuários com mais de 55%
    result_dt = result_dt[result_dt[var] > 0.55]
    group = result_dt.groupby(['context']).count()['uid']
    fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
    sns.catplot(x="context", kind="count", palette="ch:.25", data=result_dt, ax=ax)
    plt.show()

def chart_comp_pattern_obs(result_dt,vars=[]):
    count =0
    plt.figure(figsize=(30, 20), dpi=100)

    for ctx in days:
        context_dt = result_dt.xs(ctx, level=1, axis=0, drop_level=True)[vars]
        count += 1
        plt.subplot(7, 1, count)
        sns.lineplot(data=context_dt, palette="tab10", linewidth=2.5)
        plt.legend(bbox_to_anchor=(1.08, 1), loc=1, borderaxespad=0.)
        plt.title('Context: {}'.format(ctx))
        plt.grid(True)

    plt.show()


coll_stast = 'm_day', 'm_pattern', 'm_pattern_day', 'm_slide_pattern', 'm_pattern_day_obs', \
             'std_day', 'std_pattern', 'std_pattern_day', 'std_slide_pattern', 'std_pattern_day_obs', \
                'md_day', 'md_pattern', 'md_pattern_day', 'md_slide_pattern', 'md_pattern_day_obs'

#fluxo de similaridade intra variáveis por contexto
#chart_mult_var(result_dt,'m_pattern_day2')
#chart_mult_var(result_dt,'m_pattern3')
#chart_mult_var(result_dt,'m_slide_pattern')
#chart_mult_var(result_dt,'m_pattern_day_obs')
#chart_mult_var(result_dt,'m_pattern_day2')
#chart_mult_var(result_dt,'m_pattern_day3')

#Gráfico de comparação entre predições dos padrões
chart_comp_pattern_obs(result_dt,[ 'm_pattern_day1','m_pattern_day2', 'm_pattern_day3','m_pattern_day4'])

#comparação entre padrões sem e cosiderando número mínimo de observações
chart_comp_pattern_obs(result_dt,[ 'm_pattern_day2', 'm_pattern_day3','m_pattern_day_obs'])

#similaridade entre dias
chart_mult_var(result_dt,'m_day')

#Relação entre padrões e dias seguintes
chart_pattern_rot(result_dt, 'm_pattern_day2')
chart_pattern_rot(result_dt, 'm_pattern_day3')
#chart_ptn_day_high(result_dt, 'm_pattern_day2')


#Gráfico de comparação entre padrões
chart_comp_pattern_obs(result_dt,[ 'm_pattern1','m_pattern2', 'm_pattern3','m_pattern4'])


print('Correlação entre rotina e predição dos padrões (1week, 2week, 3week, 4week)\n')
print(result_dt[['m_day', 'm_pattern_day1']].corr())
print(result_dt[['m_day', 'm_pattern_day2']].corr())
print(result_dt[['m_day', 'm_pattern_day3']].corr())
print(result_dt[['m_day', 'm_pattern_day4']].corr())

print('------- Média da predição dos dias através dos padrões\n')
result_pattern_day_dt = result_dt[['m_pattern_day1', 'm_pattern_day2', 'm_pattern_day3', 'm_pattern_day4']]
print(result_pattern_day_dt.mean())


print('---------- média de similaridade entre padrões\n ')
result_pattern_dt = result_dt[['m_pattern1', 'm_pattern2', 'm_pattern3', 'm_pattern4']]
print(result_pattern_dt.mean())

print('------- Média padrões considerando observações\n')
result_pattern_day_dt = result_dt[['m_pattern_day2', 'm_pattern_day3', 'm_pattern_day_obs']]
print(result_pattern_day_dt.mean())





