# import required modules


import pandas as pd
import numpy as np
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

import matplotlib.pyplot as plt
import seaborn as sns
import re, string, math
from datetime import datetime
#import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
  fixed name; str
  
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
  return a/b if b else 999


@st.experimental_memo(suppress_st_warning=True)
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

    all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/128/results.csv?api_key=KisyFBTEg3GfiTZbrly189LJAHjwAzFIW7l9UElB", parse_dates = ['date','appointment_date','date_confirmed','date_cancelled'])
    all_data.loc[:,'date'] = pd.to_datetime(all_data.loc[:,'date'])
    # rename columns
    all_data = all_data.rename(columns={'year': 'model_year', 'name':'status'})
    all_data.loc[:, 'model_year'] = all_data.loc[:,'model_year'].apply(lambda x: 'XX' if math.isnan(x) else str(int(x)))
    all_data.loc[:,'brand'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['brand']).upper(), axis=1)
    all_data.loc[:,'model'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['model']).upper(), axis=1)

    # remove cancelled transactions
    all_data = all_data[all_data['status']!='Cancelled']
    # remove duplicates and fix names
    all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']), axis=1)
    all_data.loc[:, 'model/year'] = all_data.loc[:, 'model'].str.upper() + '/' + all_data.loc[:, 'model_year']
    all_data.loc[:, 'plate_number'] = all_data.plate_number.fillna('0000000').apply(lambda x: x[:3].upper() + x[3:].strip())
    
    # desired columns
    cols = ['id', 'date', 'email','full_name','brand', 'model', 'model_year', 
        'appointment_date', 'mechanic_name', 'sub_total', 'service_fee', 'total_cost', 
        'date_confirmed', 'status', 'status_of_payment','customer_id','fuel_type',
        'transmission','plate_number', 'phone','address','mileage','model/year']
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

#@st.experimental_memo(suppress_st_warning=True)
def cohort_analysis(df_data):
    '''
    Parameters
    ----------
    df_data : dataframe
        Customer transaction data
    
    Returns
    -------
    cohort_pivot : pivot table
        Data for cohort analysis chart
    
    '''
    
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
    sns.heatmap(cohort_pivot, annot=True, fmt='.1%', mask=cohort_pivot.isnull(), 
                square=True, linewidths=.5, cmap=sns.cubehelix_palette(8), annot_kws={"fontsize":15},
                vmin=0, vmax=0.1)
    plt.xlabel('Months After First Purchase', size=18)
    plt.ylabel('First Purchase Cohort', size=18)
    plt.title('Cohort Analysis')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    return cohort_pivot

@st.experimental_memo(suppress_st_warning=True)
def cohort_rfm(df):
    '''
    
    Parameters
    ----------
    df : dataframe
        Prepared customer transaction dataframe

    Returns
    -------
    df_retention : dataframe
        Customer dataframe with extracted RFM features

    '''
    df_retention = df.groupby('full_name').agg(
                                       cohort=('date', lambda x: x.min().year*100 + x.min().month),
                                       recency=('date', lambda x: (x.max() - x.min()).days),
                                       frequency=('id', lambda x: len(x) - 1),
                                       total_sales=('total_cost', lambda x: round(np.sum(x), 2)),
                                       avg_sales=('total_cost', lambda x: round(np.mean(x), 2)),
                                       T = ('date', lambda x: (datetime.today()-x.min()).days + 1),
                                       year=('date', lambda x: x.min().year),
                                       month=('date', lambda x: x.min().month),
                                       )
    df_retention.columns = ['cohort', 'recency', 'frequency', 'total_sales', 
                         'avg_sales', 'T', 'year', 'month']
    df_retention.loc[:,'ITT'] = df_retention.apply(lambda row: round(get_ratio(row['recency'], row['frequency']), 2), axis=1)
    df_retention.loc[:, 'last_txn'] = df_retention.apply(lambda x: int(x['T'] - x['recency']), axis=1)
    df_retention = df_retention.fillna(0)
    # filter data by returning customers
    df_retention = df_retention[df_retention['avg_sales'] > 0]
    
    return df_retention

#@st.experimental_memo(suppress_st_warning=True)
def customer_lv(df_retention):
    '''
    Calculates customer lifetime value

    Parameters
    ----------
    df_retention : dataframe
        Cohort rfm data

    Returns
    -------
    customer_lv : dataframe
        Customer lifetime value and its components

    '''
    
    monthly_clv, avg_sales, purchase_freq, churn = list(), list(), list(), list()

    # calculate monthly customer lifetime value per cohort
    for d in sorted(df_retention['cohort'].unique()):
      customer_m = df_retention[df_retention['cohort']==d]
      avg_sales.append(round(np.mean(customer_m['avg_sales']), 2))
      purchase_freq.append(round(np.mean(customer_m['frequency']), 2))
      retention_rate = customer_m[customer_m['frequency']>0].shape[0]/customer_m.shape[0]
      churn.append(round(1-retention_rate,2))
      clv = round((avg_sales[-1]*purchase_freq[-1]/churn[-1]), 2)
      monthly_clv.append(clv)
    
    customer_lv = pd.DataFrame({'cohort':sorted(df_retention['cohort'].unique()), 'clv':monthly_clv, 
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

def customer_lv_(df_retention):
    '''
    Calculates customer lifetime value

    Parameters
    ----------
    df_retention : dataframe
        Cohort rfm data

    Returns
    -------
    customer_lv : dataframe
        Customer lifetime value and its components

    '''
    
    monthly_clv, avg_sales, purchase_freq, churn = list(), list(), list(), list()

    # calculate monthly customer lifetime value per cohort
    for d in sorted(df_retention['cohort'].unique()):
      customer_m = df_retention[df_retention['cohort']==d]
      avg_sales.append(round(np.mean(customer_m['avg_sales']), 2))
      purchase_freq.append(round(np.mean(customer_m['frequency']), 2))
      retention_rate = customer_m[customer_m['frequency']>0].shape[0]/customer_m.shape[0]
      churn.append(round(1-retention_rate,2))
      clv = round((avg_sales[-1]*purchase_freq[-1]/churn[-1]), 2)
      monthly_clv.append(clv)
    
    customer_lv = pd.DataFrame({'cohort':sorted(df_retention['cohort'].unique()), 'clv':monthly_clv, 
                                 'avg_sales': avg_sales, 'purchase_freq': purchase_freq,
                                 'churn': churn})
    
    cohorts = sorted(customer_lv.cohort.unique())
    data = [customer_lv.clv, customer_lv.churn, customer_lv.avg_sales, 
            customer_lv.purchase_freq]
    y_labels = ['CLV (Php)', 'Churn %', 'Avg Sales (Php)', 'Purchase Freq.']
    
    fig = make_subplots(rows=len(data), cols=1, 
                        shared_xaxes=True, vertical_spacing=0.02)
    
    for i, col in enumerate(data, start=1):
        fig.add_trace(go.Scatter(x=cohorts, y=data[i-1],
                                 line = dict(width=4, dash='dash'),
                                 name= y_labels[i-1]),
                         row=i, col=1)
        fig.update_yaxes(title_text = y_labels[i-1], row=i, col=1)
    fig.update_xaxes(type='category')

    fig.update_layout(title_text = 'Cohort CLV characteristics',
                      height = 1200,
                      width = 800)
    st.plotly_chart(fig)

#@st.experimental_memo(suppress_st_warning=True)
def bar_plot(df_retention, option = 'Inter-transaction time (ITT)'):
    '''
    Plots inter-transaction time of returning customers

    Parameters
    ----------
    df_retention : dataframe

    Returns
    -------
    ITT plot

    '''
    choice = {'Inter-transaction time (ITT)': 'ITT',
              'Average Sales': 'avg_sales',
              'Predicted No. of Transactions': 'expected_purchases',
              'Predicted Sales': 'pred_sales',
              'Active Probability': 'prob_active'}
    
    bins = st.slider('Bins: ', 5, 50, 
                     value=25,
                     step=5)
    
    a = df_retention[df_retention['frequency'] == 1][choice[option]]
    b = df_retention[df_retention['frequency'] > 1][choice[option]]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = a, nbinsx=bins, name='Single Repeat'))
    fig.add_trace(go.Histogram(x = b, nbinsy=bins, name='Multiple Repeat'))
    
    x_lab = {'Inter-transaction time (ITT)': 'Days',
             'Average Sales': 'Amount (Php)',
             'Predicted No. of Transactions': 'Count',
             'Predicted Sales': 'Amount (Php)',
             'Active Probability': '%'}
    
    fig.update_layout(barmode='overlay',
                      xaxis_title_text=x_lab[option],
                      yaxis_title_text='Number of customers',
                      title_text='{} by returning customers'.format(option))
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

@st.experimental_singleton(suppress_st_warning=True)
def fit_models(df_retention):
    pnbd = ParetoNBDFitter(penalizer_coef=0.001)
    pnbd.fit(df_retention['frequency'], df_retention['recency'], df_retention['T'])
    # model to estimate average monetary value of customer transactions
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    # filter df to returning customers
    returning_df_retention = df_retention[df_retention['frequency']>0]
    # fit model
    ggf.fit(returning_df_retention['frequency'], returning_df_retention['avg_sales'])
    
    return pnbd, ggf

def plot_prob_active(_pnbd):
    '''
    Plots the active probability matrix for the range of recency and frequency
    
    Parameters
    ----------
    pnbd : model
        Fitted Pareto/NBD model
    '''
    st.title('Active Probability Matrix')
    st.markdown('''
                High recency means customer is most likely still active (long intervals between purchases).\n
                High frequency with low recency means one-time instance of many orders with long hiatus.
                ''')
    fig = plt.figure(figsize=(12,8))
    plot_probability_alive_matrix(_pnbd)
    st.pyplot(fig)

@st.experimental_memo(suppress_st_warning=True)
def update_retention(_pnbd, _ggf, t, df_retention):
    # calculate probability of active
    df_retention.loc[:,'prob_active'] = df_retention.apply(lambda x: 
           _pnbd.conditional_probability_alive(x['frequency'], x['recency'], x['T']), 1)
    df_retention.loc[:, 'expected_purchases'] = df_retention.apply(lambda x: 
            _pnbd.conditional_expected_number_of_purchases_up_to_time(t, x['frequency'], x['recency'], x['T']),1)
    df_retention.loc[:, 'prob_1_purchase'] = df_retention.apply(lambda x: 
            _pnbd.conditional_probability_of_n_purchases_up_to_time(1, t, x['frequency'], x['recency'], x['T']),1)
    # predicted average sales per customer
    df_retention.loc[:, 'pred_avg_sales'] = _ggf.conditional_expected_average_profit(df_retention['frequency'],df_retention['avg_sales'])
    # clean negative avg sales output from model
    df_retention.loc[:,'pred_avg_sales'][df_retention.loc[:,'pred_avg_sales'] < 0] = 0
    # calculated clv for time t
    df_retention.loc[:,'pred_sales'] = df_retention.apply(lambda x: 
            x['expected_purchases'] * x['avg_sales'], axis=1)
    return df_retention.round(3)
        
@st.experimental_memo(suppress_st_warning=True)
def search_for_name_retention(name, df_retention):
    '''
    Function to search for customer names in backend data
    '''
    df_retention = df_retention.reset_index()
    # lower to match with names in dataframe
    df_retention.loc[:,'full_name'] = df_retention.apply(lambda x: x['full_name'].lower(), axis=1)
    # search row with name
    names_retention = df_retention[df_retention.apply(lambda x: name.lower() in x['full_name'], axis=1)]
    df_temp_retention = names_retention[['full_name', 'phone', 'brand', 'model', 'address', 'prob_active', 'expected_purchases', 
                                         'avg_sales', 'pred_sales', 'last_txn', 'ITT', 'total_sales', 'cohort']]
    df_temp_retention.loc[:, 'full_name'] = df_temp_retention.loc[:, 'full_name'].str.title()
    # round off all columns except cohort
    round_cols = ['prob_active', 'expected_purchases','avg_sales', 'pred_sales', 'last_txn', 'ITT', 'total_sales']
    df_temp_retention.loc[:, round_cols] = df_temp_retention.loc[:, round_cols].round(3)
    df_temp_retention = df_temp_retention.set_index('full_name')
    return df_temp_retention

def customer_search(df_data, df_retention):
    '''
    Displays retention info of selected customers.

    Parameters
    ----------
    df_data : dataframe
    df_retention : dataframe
    models : list
        list of fitted Pareto/NBD and Gamma Gamma function

    Returns
    -------
    df_retention : dataframe
        df_retention with updated values

    '''
    # Reprocess dataframe entries to be displayed
    df_temp = df_data.reset_index().drop_duplicates(subset=['full_name', 'brand', 'model'], keep='first')[['full_name', 'phone', 'brand', 'model', 'address']]
    
    df_temp_ret = df_retention.reset_index()[['full_name', 'prob_active', 'expected_purchases', 
                                     'avg_sales', 'pred_sales', 'last_txn', 'ITT', 'total_sales', 'cohort']]
    
    df_merged = pd.merge(df_temp, df_temp_ret, how='left', left_on='full_name', right_on='full_name')
    # Capitalize first letter of each name
    df_merged.loc[:, 'full_name'] = df_merged.loc[:, 'full_name'].str.title()
    
    # table settings
    df_display = df_merged.sort_values(by='full_name')
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_column('full_name', headerCheckboxSelection = True)
    gridOptions = gb.build()
    
    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        autoSizeColumn = 'full_name',
        fit_columns_on_grid_load=False,
        theme='blue', #Add theme color to the table
        enable_enterprise_modules=True,
        height=400, 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        # row/s are selected
        
        df_list_retention = [search_for_name_retention(selected[checked_items]['full_name'], df_merged) 
                             for checked_items in range(len(selected))]
        
        df_list_retention = pd.concat(df_list_retention)
        st.dataframe(df_list_retention)          

    else:
        st.write('Click on an entry in the table to display customer data.')
        df_list_retention = pd.DataFrame()
        
    return df_list_retention

@st.experimental_memo
def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent recomputation on every rerun.
    return df.to_csv().encode('utf-8')

if __name__ == '__main__':
    st.title('MechaniGO.ph Customer Retention')
    # import data and preparation
    df_data = get_data()
    # calculates cohort rfm data
    df_retention = cohort_rfm(df_data)
    
    # fit pareto/nbd and gamma gamma models
    pnbd, ggf = fit_models(df_retention)
    
    # adjust time window
    time = st.slider('Future time window:', 15, 60, 
                     value=30,
                     step=15)
    # update df_retention with pareto results
    df_retention = update_retention(pnbd, ggf, time, df_retention)
   
    st.markdown("""
            This app searches for the **name** or **email** you select on the table.\n
            Filter the name/email on the dropdown menu as you hover on the column names. 
            Click on the entry to display data below. 
            """)
    customer_retention_list = customer_search(df_data, df_retention)
    
    if len(customer_retention_list):
        st.download_button(
            label ="Download customer data",
            data = convert_csv(customer_retention_list),
            file_name = "customer_retention.csv",
            key='download-retention-csv'
            )
    
    st.markdown('''
                Variable meanings: \n
                \n    
                - **total/avg_sales**: Total/Average sales of each customer transaction. \n
                - **ITT**: Inter-transaction time (average time between transactions). \n
                - **last_txn**: Days since last transaction. \n
                - **prob_active**: Probability that customer will still make a transaction in the future. \n
                - **expected_purchases**: Predicted no. of purchases within time t. \n
                - **pred_sales**: Predicted customer sales within time t. \n
                - **cohort**: Year-month of first customer purchase
                ''')
    
    # histogram plots customer info 
    st.write('''
             This bar plot shows the distribution of single/multiple repeat 
             transaction(s) based on:
             ''')
    option = st.selectbox('Variable to show: ', 
                          ('Inter-transaction time (ITT)', 'Average Sales', 
                           'Predicted No. of Transactions',
                           'Predicted Sales',
                           'Active Probability'))
    bar_plot(df_retention, option=option)
    
    # plot_prob_active(pnbd)
    
    st.title('Cohort Analysis')
    # plot cohort_retention_chart
    st.write('''This chart shows the retention rate for customers of various cohorts
             (grouped by first month of transaction). The data shows the percentage 
             of customers in that cohort that are retained months after their initial 
             purchase.''')
    cohort_pivot = cohort_analysis(df_data)
    
    st.title('Cohort Lifetime Value')
    # calculates customer rfm data and clv
    st.write('''
             These plots show the CLV for each cohort and how the trend of each 
             of its components (frequency, average total sales, churn%) vary.
             ''')
    clv = customer_lv_(df_retention)
    
    
    