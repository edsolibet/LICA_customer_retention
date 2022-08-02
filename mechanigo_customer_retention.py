# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:03:31 2022

@author: carlo
"""
# import required modules

import sys
import subprocess
import pkg_resources

required = {'pandas', 'numpy', 'lifetimes', 'matplotlib', 'seaborn', 'statsmodels', 
            'datetime', 'joblib', 'streamlit', 'plotly', 'streamlit-aggrid'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


import pandas as pd
import numpy as np
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels, re, string, math
from datetime import datetime
from joblib import dump, load
import plotly.graph_objects as go
import plotly.express as px
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes import GammaGammaFitter

pd.options.mode.chained_assignment = None  # default='warn'

def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()

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
  # removes emojis and ascii characters (i.e. chinese chars)
  name = remove_emoji(name).encode('ascii', 'ignore').decode()
  # split name by spaces
  for n in name.split(' '):
    if n not in name_list:
    # check each character for punctuations minus '.' and ','
      name_list.append(''.join([ch for ch in n 
                                if ch not in string.punctuation.replace('.', '')]))
    else:
        continue
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
    #all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/103/results.csv?api_key=QHb7Vxu8oKMyOVhf4bw7YtWRcuQfzvMS6YBSqgeM")
    all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/128/results.csv?api_key=KisyFBTEg3GfiTZbrly189LJAHjwAzFIW7l9UElB", parse_dates = ['date','appointment_date','date_confirmed','date_cancelled'])
    all_data.loc[:,'date'] = pd.to_datetime(all_data.loc[:,'date'])
    # rename columns
    all_data = all_data.rename(columns={'year': 'model_year', 'name':'status'})
    all_data.loc[:, 'model_year'] = all_data.loc[:,'model_year'].apply(lambda x: 'XX' if math.isnan(x) else str(int(x)))
    all_data.loc[:,'brand'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['brand']).upper(), axis=1)
    all_data.loc[:,'model'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['model']).upper(), axis=1)
    #all_data.loc[:,'month'] = all_data.apply(lambda x: x['date'].month, 1)
    #all_data.loc[:,'year'] = all_data.apply(lambda x: x['date'].year, 1)
    # remove cancelled transactions
    all_data = all_data[all_data['status']!='Cancelled']
    # remove duplicates and fix names
    all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']), axis=1)
    all_data['model/year'] =all_data['model_year']+'/' + all_data['model'].str.upper()
    # desired columns
    cols = ['id', 'date', 'email','full_name','brand', 'model', 'model_year', 
        'appointment_date', 'mechanic_name', 'sub_total', 'service_fee', 'total_cost', 
        'date_confirmed', 'status', 'status_of_payment','customer_id','fuel_type','transmission','plate_number', 'phone','address','mileage','model/year']
    # columns used for dropping duplicates
    drop_subset = ['full_name', 'brand', 'model', 'appointment_date','customer_id']
    all_data_ = all_data[cols].drop_duplicates(subset=drop_subset, keep='first')
    # combine "service name" of entries with same transaction id
    temp = all_data.fillna('').groupby(['id','full_name'])['service_name'].apply(lambda x: fix_name(', '.join(x).lower())).sort_index(ascending=False).reset_index()
    # merge dataframes
    df_data = all_data_.merge(temp, left_on=['id', 'full_name'], right_on=['id','full_name'])
    # convert date to datetime
    df_data.loc[:,'date'] = pd.to_datetime(df_data.loc[:,'date'])
    
    # converts each row date to a cohort
    df_data.loc[:,'cohort'] = df_data.apply(lambda row: row['date'].year*100 + row['date'].month, axis=1)
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
                                       avg_sales=('total_cost', lambda x: np.mean(x)),
                                       T = ('date', lambda x: (datetime.today()-x.min()).days + 1),
                                       year=('date', lambda x: x.min().year),
                                       month=('date', lambda x: x.min().month),
                                       )
    df_cohort.columns = ['cohort', 'recency', 'frequency', 'total_sales', 
                         'avg_sales', 'T', 'year', 'month']
    df_cohort.loc[:,'ITT'] = df_cohort.apply(lambda row: round(get_ratio(row['recency'], row['frequency']), 2), axis=1)
    df_cohort.loc[:, 'last_txn'] = df_cohort.apply(lambda x: x['T'] - x['recency'], axis=1)
    df_cohort = df_cohort.fillna(0)
    # filter data by returning customers
    df_cohort = df_cohort[df_cohort['avg_sales'] > 0]
    
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
      avg_sales.append(round(np.mean(customer_m['avg_sales']), 2))
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
    plt.title('Cohort Lifetime Value', fontsize=14);
    # set secondary y-axis
    ax2 = ax1[0].twinx()
    ax2.plot(range(len(customer_lv_.cohort)), customer_lv_.churn, '--o', color=colors[0][1])
    ax2.set_ylim([0.5, 1])
    ax2.set_ylabel('churn %', color=colors[0][1], fontsize=12)
    # set shared x-label and x-ticks
    ax2.set_xticks(range(len(customer_lv_.cohort)))
    ax2.set_xticklabels(customer_lv_.cohort)
    ax2.set_xlabel('first_cohort', fontsize=12);
    
    ax1[1].plot(range(len(customer_lv_.cohort)), customer_lv_.avg_sales, '--o', color=colors[1][0]);
    #ax1[1].set_ylim([0, round(customer_clv.clv.max()*1.2)])
    ax1[1].set_ylabel('avg sales', color=colors[1][0], fontsize=12)
    # set secondary y-axis
    ax2 = ax1[1].twinx()
    ax2.plot(range(len(customer_lv_.cohort)), customer_lv_.purchase_freq, '--o', color=colors[1][1])
    #ax2.set_ylim([0.5, 1])
    ax2.set_ylabel('purchase freq', color=colors[1][1], fontsize=12)
    # set shared x-label and x-ticks
    ax2.set_xticks(range(len(customer_lv_.cohort)))
    ax2.set_xticklabels(customer_lv_.cohort)
    fig.autofmt_xdate(rotation=90, ha='right')
    ax2.set_xlabel('first_cohort', fontsize=12);
    
    st.pyplot(fig)
    return customer_lv

def bar_plot(df_cohort, option = 'Inter-transaction time'):
    '''
    Plots inter-transaction time of returning customers

    Parameters
    ----------
    df_cohort : dataframe

    Returns
    -------
    ITT plot

    '''
    option = st.selectbox('Variable to show: ', 
                          ('Inter-transaction time', 'Average Sales', 
                           'Predicted Average Sale', 'Predicted CLV',
                           'Active Probability'))
    choice = {'Inter-transaction time': 'ITT',
              'Average Sales': 'avg_sales',
              'Predicted Average Sale': 'pred_avg_sale',
              'Predicted CLV': 'pred_clv',
              'Active Probability': 'prob_alive'}
    fig, ax1 = plt.subplots()
    bins = st.slider('Bins: ', 5, 55, 
                     value=25,
                     step=5)
    a = df_cohort[df_cohort['frequency'] == 1][choice[option]]
    b = df_cohort[df_cohort['frequency'] > 1][choice[option]]
    ax1.hist([a.values, b.values], bins=bins, label=['Single', 'Multiple'])
    x_lab = {'Inter-transaction time': 'Days',
             'Average Sales': 'Amount (Php)',
             'Predicted Average Sale': 'Amount (Php)',
             'Predicted CLV': 'Amount (Php)',
             'Active Probability': '%'}
    plt.xlabel(x_lab[option], fontsize=12)
    plt.ylabel('Number of customers', fontsize=12)
    plt.title('{} by returning customers'.format(option), fontsize=14);
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

def fit_models(df_cohort):
    pnbd = ParetoNBDFitter(penalizer_coef=0.001)
    pnbd.fit(df_cohort['frequency'], df_cohort['recency'], df_cohort['T'])
    # model to estimate average monetary value of customer transactions
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    # filter df to returning customers
    returning_df_cohort = df_cohort[df_cohort['frequency']>0]
    # fit model
    ggf.fit(returning_df_cohort['frequency'], returning_df_cohort['avg_sales'])
    
    return pnbd, ggf

def update_cohort(pnbd, ggf, t, df_cohort):
    # calculate probability of active
    df_cohort.loc[:,'prob_alive'] = df_cohort.apply(lambda x: 
            pnbd.conditional_probability_alive(x['frequency'], x['recency'], x['T']), 1)
    df_cohort.loc[:, 'expected_purchases'] = df_cohort.apply(lambda x: 
            pnbd.conditional_expected_number_of_purchases_up_to_time(t, x['frequency'], x['recency'], x['T']),1)
    df_cohort.loc[:, 'prob_1_purchase'] = df_cohort.apply(lambda x: 
            pnbd.conditional_probability_of_n_purchases_up_to_time(1, t, x['frequency'], x['recency'], x['T']),1)
    # predicted average sales per customer
    df_cohort.loc[:, 'pred_avg_sales'] = ggf.conditional_expected_average_profit(df_cohort['frequency'],df_cohort['total_sales'])
    # clean negative avg sales output from model
    df_cohort.loc[:,'pred_avg_sales'][df_cohort.loc[:,'pred_avg_sales'] < 0] = 0
    # calculated clv for time t
    df_cohort.loc[:,'pred_clv'] = df_cohort.apply(lambda x: 
            x['expected_purchases'] * x['pred_avg_sales'], axis=1)
    return df_cohort
        
def pareto_NBD_model(df_cohort, t):
    '''
    Fits Pareto/Non Binomial Distribution (NBD) model on RFM data

    Parameters
    ----------
    df_cohort : dataframe
        See above
    t : integer
        Fture time interval (days) to predict

    Returns
    -------
    df_cohort : dataframe
        modified df_cohort
    pnbd : model
        prediction model

    '''
    pnbd = ParetoNBDFitter(penalizer_coef=0.001)
    pnbd.fit(df_cohort['frequency'], df_cohort['recency'], df_cohort['T'])
    # calculate probability of active
    df_cohort.loc[:,'prob_alive'] = df_cohort.apply(lambda x: 
            pnbd.conditional_probability_alive(x['frequency'], x['recency'], x['T']), 1)
    df_cohort.loc[:, 'expected_purchases'] = df_cohort.apply(lambda x: 
            pnbd.conditional_expected_number_of_purchases_up_to_time(t, x['frequency'], x['recency'], x['T']),1)
    df_cohort.loc[:, 'prob_1_purchase'] = df_cohort.apply(lambda x: 
            pnbd.conditional_probability_of_n_purchases_up_to_time(1, t, x['frequency'], x['recency'], x['T']),1)
    
    return df_cohort, pnbd

def gamma_gamma_model(df_cohort):
    '''
    Fits Gamma Gamma model to predict avg monetary value and CLV

    Parameters
    ----------
    df_cohort : dataframe
        See above

    Returns
    -------
    df_cohort : dataframe
        Modified df_cohort
    ggf : model
        Gamma gamma model

    '''
    # model to estimate average monetary value of customer transactions
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    # filter df to returning customers
    returning_df_cohort = df_cohort[df_cohort['frequency']>0]
    # fit model
    ggf.fit(returning_df_cohort['frequency'], returning_df_cohort['avg_sales'])
    # predicted average sales per customer
    df_cohort.loc[:, 'pred_avg_sales'] = ggf.conditional_expected_average_profit(df_cohort['frequency'],df_cohort['total_sales'])
    # clean negative avg sales output from model
    df_cohort.loc[:,'pred_avg_sales'][df_cohort.loc[:,'pred_avg_sales'] < 0] = 0
    # calculated clv for time t
    df_cohort.loc[:,'pred_clv'] = df_cohort.apply(lambda x: 
            x['expected_purchases'] * x['pred_avg_sales'], axis=1)
    return df_cohort, ggf

def search_for_name(name, df_data):
  df_data.full_name = df_data.apply(lambda x: x['full_name'].lower(), axis=1)
  names = df_data[df_data.apply(lambda x: name.lower() in x['full_name'], axis=1)]
  df_temp = names[['customer_id','full_name', 'brand', 'model/year','fuel_type',
                   'transmission','plate_number','phone','address','mileage',
                   'appointment_date','id','service_name']]
  df_temp['full_name'] = df_temp['full_name'].str.title()
  return df_temp.set_index('full_name')

def search_for_name_retention(name, df_cohort):
    df_cohort = df_cohort.reset_index()
    df_cohort.loc[:,'full_name'] = df_cohort.apply(lambda x: x['full_name'].lower(), axis=1)
    names_retention = df_cohort[df_cohort.apply(lambda x: name.lower() in x['full_name'], axis=1)]
    df_temp_retention = names_retention[['full_name', 'recency', 'frequency', 'T', 
                                       'total_sales', 'avg_sales', 'ITT', 'last_txn',
                                       'prob_alive', 'expected_purchases', 'prob_1_purchase',
                                       'pred_avg_sales', 'pred_clv']]
    df_temp_retention['full_name'] = df_temp_retention['full_name'].str.title()
    return df_temp_retention.set_index('full_name')

def customer_search(df_data, df_cohort, models):
    # Reprocess dataframe entries to be displayed
    df_temp = df_data.reset_index()[['full_name','email']].drop_duplicates(subset=['full_name','email'], keep='first')
    df_temp['full_name'] = df_temp['full_name'].str.title()
    df_display = df_temp
    
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()
    
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        theme='blue', #Add theme color to the table
        enable_enterprise_modules=True,
        height=200, 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        df_list_data, df_list_retention = list(), list()
        for checked_items in range(len(selected)):
            df_list_data.append(search_for_name(selected[checked_items]['full_name'], df_data))
        st.dataframe(pd.concat(df_list_data))
        st.write('Found items: ' + str(len(data_selection['selected_rows']))+
                                      '.'+' Select on the expand button (hover on data table) to display data in fullscreen.')
        time = st.slider('Probability up to what time in days:', 15, 360, 
                     value=30,
                     step=15)
        pnbd, ggf = models
        df_cohort = update_cohort(pnbd, ggf, time, df_cohort)
        for checked_items in range(len(selected)):
            df_list_retention.append(search_for_name_retention(selected[checked_items]['full_name'], df_cohort))
        st.dataframe(pd.concat(df_list_retention))
        st.write('Entries: '+str(len(pd.concat(df_list_retention))))
    else:
        st.write('Please click on an entry in the table to display data.')

if __name__ == '__main__':
    st.title('MechaniGo Customer Retention')
    # import data and preparation
    df_data = get_data()
    # calculates cohort rfm data
    df_cohort = cohort_rfm(df_data)
    # fit pareto/nbd and gamma gamma models
    pnbd, ggf = fit_models(df_cohort)
   
    st.markdown("""
            This app searches for the **name** or **email** you select on the table!\n
            Filter the name/email on the dropdown menu as you hover on the column names. 
            Click on the entry to display data below
            """)
    customer_search(df_data, df_cohort, [pnbd, ggf])
    
    # plot cohort_retention_chart
    st.write('''This chart shows the retention for customers of various cohorts
             (grouped by first month of transaction). The data shows the percentage 
             of customers in that cohort that are retained months after their initial 
             purchase.''')
    cohort_pivot = cohort_analysis(df_data)
    
    # calculates customer rfm data and clv
    st.write('''
             These plots show the CLV for each cohort and how the trend of each 
             of its components (frequency, average total sales, churn%).
             ''')
    clv = customer_lv(df_cohort)
    
    # plots ITT
    st.write('''
             This bar plot shows the distribution of single/multiple repeat 
             transaction(s) based on:
             ''')
    bar_plot(df_cohort)
    