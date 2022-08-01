# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:03:31 2022

@author: carlo
"""
# import required modules

import sys
import subprocess
import pkg_resources

required = {'pandas', 'numpy', 'lifetimes', 'matplotlib', 'seaborn', 'statsmodels', 'datetime', 'joblib', 'streamlit'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
from datetime import datetime
from joblib import dump, load
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes import GammaGammaFitter

def fix_name(name):
  '''
  Fix names which are duplicated.
  Ex. "John Smith John Smith"
  
  Parameters:
  -----------
    name: str
    
  Returns:
  --------
    - fixed name; str
  '''
  name_list = list()
  for n in name.split(' '):
    if n not in name_list:
      name_list.append(n)
  return ' '.join(name_list).strip()

def get_ratio(a, b):
  try:
    return a/b
  except:
    return 999

def get_data():
    
    '''
    Import data from redash query
    Perform necessary data preparations
    
    Parameters
    ----------
    None.

    Returns
    -------
    df_data: dataframe
        

    '''
    all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/103/results.csv?api_key=QHb7Vxu8oKMyOVhf4bw7YtWRcuQfzvMS6YBSqgeM")
    all_data.loc[:,'date'] = pd.to_datetime(all_data.loc[:,'date'])
    # rename columns
    all_data = all_data.rename(columns={'year': 'model_year', 'name':'status'})
    all_data.loc[:,'model_year'] = all_data.loc[:,'model_year'].fillna(0).astype(int)
    all_data.loc[:,'month'] = all_data.apply(lambda x: x['date'].month, 1)
    all_data.loc[:,'year'] = all_data.apply(lambda x: x['date'].year, 1)
    # remove cancelled transactions
    all_data = all_data[all_data['status']!='Cancelled']
    # remove duplicates and fix names
    all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']), axis=1)
    
    # desired columns
    cols = ['id', 'date', 'year', 'month', 'full_name','make', 'model', 'model_year', 
            'appointment_date', 'mechanic_name', 'sub_total', 'service_fee', 'total_cost', 
            'date_confirmed', 'status', 'status_of_payment']
    # columns used for dropping duplicates
    drop_subset = ['full_name', 'make', 'model', 'appointment_date']
    all_data_ = all_data[cols].drop_duplicates(subset=drop_subset, keep='first')
    # combine "service name" of entries with same transaction id
    temp = all_data.fillna('').groupby(['id','full_name'])['service_name'].apply(lambda x: fix_name(', '.join(x).lower())).sort_index(ascending=False).reset_index()
    # merge dataframes
    df_data = all_data_.merge(temp, left_on=['id', 'full_name'], right_on=['id','full_name'])
    # convert date to datetime
    df_data.loc[:,'date'] = pd.to_datetime(df_data.loc[:,'date'])
    
    # converts each row date to a cohort
    df_data.loc[:,'cohort'] = df_data.apply(lambda row: row['year']*100 + row['month'], axis=1)
    # get first month & year of first purchase per full_name
    cohorts = df_data.groupby('full_name')['cohort'].min().reset_index()
    cohorts.columns = ['full_name', 'first_cohort']
    # combines cohort and first_cohort
    df_data = df_data.merge(cohorts, on='full_name', how='left')
    # remove test entries
    remove_entries = ['mechanigo.ph', 'frig_test', 'sample quotation']
    df_data = df_data[df_data.loc[:,'full_name'].isin(remove_entries) == False]
    return df_data

def cohort_analysis(df):
    
    headers = df_data['cohort'].value_counts().reset_index()
    headers.columns = ['cohort', 'count']
    headers = headers.sort_values('cohort')['cohort'].to_list()
    # calculate cohort distance from difference in index in headers list between
    # cohort and first_cohort
    df_data.loc[:,'cohort_distance'] = df_data.apply(lambda row: (headers.index(row['cohort']) - headers.index(row['first_cohort'])) if (row['first_cohort'] != 0 and row['cohort'] != 0) else np.nan, axis=1)
    # pivot table to calculate amount of unique customers in each cohort based on 
    # purchase distance from first month of purchase
    # filter out first two months
    cohort_dates = headers[2:]
    filtered_dates = df_data[df_data['first_cohort'].isin(cohort_dates)]
    cohort_pivot = pd.pivot_table(filtered_dates, index='first_cohort', columns='cohort_distance', values='full_name', aggfunc=pd.Series.nunique)
    # divide each row by the first column
    cohort_pivot = cohort_pivot.div(cohort_pivot[0],axis=0)
    
    # plot heatmap of cohort retention rate
    fig_dims = (16, 16)
    fig, ax = plt.subplots(figsize=fig_dims)
    #ax.set(xlabel='Months After First Purchase', ylabel='First Purchase Cohort', title="Cohort Analysis")
    y_labels = [str(int(head)%100) + '-' + str(int(head)/100) for head in cohort_dates]
    x_labels = range(0, len(y_labels))
    plt.yticks(ticks=cohort_dates, labels=y_labels, fontsize=15, rotation=90)
    plt.xticks(x_labels, x_labels, fontsize=15)
    # adjusted scale for colorbar via vmin/vmax
    ax = sns.heatmap(cohort_pivot, annot=True, fmt='.1%', mask=cohort_pivot.isnull(), 
                square=True, linewidths=.5, cmap=sns.cubehelix_palette(8), annot_kws={"fontsize":15},
                vmin=0, vmax=0.1)
    plt.xlabel('Months After First Purchase', size=18)
    plt.ylabel('First Purchase Cohort', size=18)
    plt.title('Cohort Analysis')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    return cohort_pivot

def cohort_rfm(df):
    '''
    

    Parameters
    ----------
    df : dataframe
        Prepared customer transaction dataframe

    Returns
    -------
    df_cohort : dataframe
        Customer dataframe with extracted RFM features

    '''
    df_cohort = df.groupby('full_name').agg(
                                       cohort=('date', lambda x: x.min().year*100 + x.min().month),
                                       recency=('date', lambda x: (x.max() - x.min()).days),
                                       frequency=('id', lambda x: len(x) - 1),
                                       total_sales=('total_cost', lambda x: np.sum(x)),
                                       T = ('date', lambda x: (datetime.today()-x.min()).days + 1),
                                       year=('date', lambda x: x.min().year),
                                       month=('date', lambda x: x.min().month),
                                       )
    df_cohort.columns = ['cohort', 'recency', 'frequency', 'total_sales', 'T', 'year', 'month']
    df_cohort.loc[:,'avg_days_between_purchase'] = df_cohort.apply(lambda row: get_ratio(row['recency'], row['frequency']), axis=1)
    df_cohort = df_cohort.fillna(0)
    # filter data by returning customers
    df_cohort = df_cohort[df_cohort['total_sales'] > 0]
    
    return df_cohort

def customer_lv(df_cohort):
    '''
    Calculates customer lifetime value

    Parameters
    ----------
    df_cohort : dataframe
        Cohort rfm data

    Returns
    -------
    customer_lv : dataframe
        Customer lifetime value and its components

    '''
    
    monthly_clv, avg_sales, purchase_freq, churn = list(), list(), list(), list()

    # calculate monthly customer lifetime value per cohort
    for d in sorted(df_cohort['cohort'].unique()):
      customer_m = df_cohort[df_cohort['cohort']==d]
      avg_sales.append(round(np.mean(customer_m['total_sales']), 2))
      purchase_freq.append(round(np.mean(customer_m['frequency']), 2))
      retention_rate = customer_m[customer_m['frequency']>0].shape[0]/customer_m.shape[0]
      churn.append(round(1-retention_rate,2))
      clv = round((avg_sales[-1]*purchase_freq[-1]/churn[-1]), 2)
      monthly_clv.append(clv)
    
    customer_lv = pd.DataFrame({'cohort':sorted(df_cohort['cohort'].unique()), 'clv':monthly_clv, 
                                 'avg_sales': avg_sales, 'purchase_freq': purchase_freq,
                                 'churn': churn})
    # plot monthly clv
    colors = [['dodgerblue', 'red'],['green', 'orange']]
    customer_lv_ = customer_lv.iloc[:,:]
    
    fig, ax1 = plt.subplots(2,1, figsize=(12, 10))
    ax1[0].plot(range(len(customer_lv_.cohort)), customer_lv_.clv, '--o', color=colors[0][0]);
    ax1[0].set_ylim([0, round(customer_lv.clv.max()*1.2)])
    ax1[0].set_ylabel('customer clv', color=colors[0][0])
    ax1[0].axhline(y=customer_lv_.clv.mean(), color='black', linestyle='--');
    # set secondary y-axis
    ax2 = ax1[0].twinx()
    ax2.plot(range(len(customer_lv_.cohort)), customer_lv_.churn, '--o', color=colors[0][1])
    ax2.set_ylim([0.5, 1])
    ax2.set_ylabel('churn %', color=colors[0][1])
    # set shared x-label and x-ticks
    ax2.set_xticks(range(len(customer_lv_.cohort)))
    ax2.set_xticklabels(customer_lv_.cohort)
    ax2.set_xlabel('first_cohort');
    
    ax1[1].plot(range(len(customer_lv_.cohort)), customer_lv_.avg_sales, '--o', color=colors[1][0]);
    #ax1[1].set_ylim([0, round(customer_clv.clv.max()*1.2)])
    ax1[1].set_ylabel('avg sales', color=colors[1][0])
    # set secondary y-axis
    ax2 = ax1[1].twinx()
    ax2.plot(range(len(customer_lv_.cohort)), customer_lv_.purchase_freq, '--o', color=colors[1][1])
    #ax2.set_ylim([0.5, 1])
    ax2.set_ylabel('purchase freq', color=colors[1][1])
    # set shared x-label and x-ticks
    ax2.set_xticks(range(len(customer_lv_.cohort)))
    ax2.set_xticklabels(customer_lv_.cohort)
    fig.autofmt_xdate(rotation=90, ha='right')
    ax2.set_xlabel('first_cohort');
    
    st.pyplot(fig)
    return df_cohort, customer_lv

if __name__ == '__main__':
    # import data and preparation
    df_data = get_data()
    # show dataframe in streamlit
    st.write('Customer transaction data in MechaniGO.ph')
    st.dataframe(df_data)
    # plot cohort_retention_chart
    st.write('''This chart shows the retention for customers of various cohorts
             (grouped by first month of transaction). The data shows the percentage 
             of customers in that cohort that are retained months after their initial 
             purchase.''')
    cohort_pivot = cohort_analysis(df_data)
    # calculates cohort rfm data
    df_cohort = cohort_rfm(df_data)
    # calculates customer rfm data and clv
    st.write('''
             These plots show the CLV for each cohort and how the trend of each 
             of its components (frequency, average total sales, churn%).
             ''')
    clv = customer_lv(df_cohort)