#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:02:21 2023

@author: kaushiknarasimha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
#from google.api_core.protobuf_helpers import get_messages
from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "*************.json"

client = bigquery.Client()

query = """
SELECT
  state_number,
  count(number_of_drunk_drivers) AS drunk_drivers_count
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
WHERE number_of_drunk_drivers > 0
GROUP BY state_number
ORDER BY
  state_number
"""

query_job=client.query(
query.format(2015),
location='US',
job_config=bigquery.QueryJobConfig(maximum_bytes_billed=50_000_000))

t=dict(query_job)
df=pd.DataFrame.from_dict(t, orient='index')
df.reset_index(inplace=True)
df.columns=['State','Drunk']
df.head()

us_accidents_sex_df = pd.read_excel('us_traffic_fatalities_per_sex.xlsx', skiprows=6, skipfooter=116, header=[1,2,3,4])
us_accidents_sex_df.head(3)



us_accidents_sex_df.columns[0]


col=us_accidents_sex_df[us_accidents_sex_df['Year', 'Unnamed: 0_level_1', 'Unnamed: 0_level_2', 'Unnamed: 0_level_3']==2015]
col


Male=col['Male (> 15 Years Old)', 'Involvement Rate', 'per 100K Licensed', 'Drivers']
print(Male)
Female=col['Female (> 15 Years Old)', 'Involvement Rate', 'per 100K Licensed', 'Drivers']
print(Female)
dic= {'Male Drivers': Male, 'Female Drivers': Female}

sex_df_2015 = pd.DataFrame(dic)
sex_df_2015


sex_df_2015.plot.bar(title='Fatal traffic crashes per 100k drivers by gender').set(xlabel='Gender',ylabel='Involvement Rate')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off


Q1 = """SELECT
  state_number,
  state_name,
  COUNT(consecutive_number) AS accidents,
  SUM(number_of_fatalities) AS fatalities,
  SUM(number_of_fatalities) / COUNT(consecutive_number) AS fatalities_per_accident
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
GROUP BY
  state_number, state_name
ORDER BY
  state_number """



query_job=client.query(
Q1.format(2016),
location='US',
job_config=bigquery.QueryJobConfig(maximum_bytes_billed=50_000_000))


#for row in query_job:
#    print(row)



a=[]
b=[]
c=[]
d=[]
e=[]
for row in query_job:
    a.append(row[0])
    b.append(row[1])
    c.append(row[2])
    d.append(row[3])
    e.append(row[4])
ddd = {'state_number': a, 'state_name': b, 'accidents': c, 'fatalities': d, 'fatalities_per_accident': e}
accidents_state_df = pd.DataFrame(ddd)


accidents_state_df.head()


# US codes dictionary
us_state_abbrev = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'Dist of Columbia': 'DC'}
# Adding codes column to the dataframe:
accidents_state_df['state_code'] = accidents_state_df['state_name'].map(us_state_abbrev)
accidents_state_df.head(3)

#######################################

import matplotlib.pyplot as plt
from matplotlib import gridspec
from plotly.offline import iplot



choropleth_map_title = 'Traffic Fatalities by State - 2016<br>(hover for breakdown)'
def plot_choropleth_df(locations_series, data_series, text_series):
    scl_blue_brwn = [[0.0, 'rgb(171,217,233)'], [0.16666666666666666, 'rgb(224,243,248)'],
                     [0.3333333333333333, 'rgb(254,224,144)'], [0.5, 'rgb(253,174,97)'],
                     [0.6666666666666666, 'rgb(244,109,67)'], [0.8333333333333334, 'rgb(215,48,39)'],
     [1.0, 'rgb(165,0,38)']]


    data = [ dict(
            type='choropleth',
            colorscale = scl_blue_brwn,
            autocolorscale = False,
            locations = locations_series,
            z = data_series,
            locationmode = 'USA-states',
            text = text_series,
            marker = dict(
                line = dict(
                    color = 'rgb(255, 255, 255)',
                    width = 2)
                ),
            colorbar = dict(
                title = "Accidents")
            ) ]

    layout = dict(
            title = choropleth_map_title,
            geo = dict(
                 scope = 'usa',
                 projection = dict(type = 'albers usa'),
                 countrycolor = 'rgb(255, 255, 255)',
                 showlakes = True,
                 lakecolor = 'rgb(255, 255, 255)')
             )

    fig = dict(data=data, layout=layout)
    iplot(fig, image='png')
    
locations_series = accidents_state_df['state_code']
data_series = accidents_state_df['fatalities']
text_series = accidents_state_df['state_name'] + '<br>' + 'Accidents: ' + accidents_state_df['accidents'].astype(str) + '<br>' + 'Fatalities: '+ accidents_state_df['fatalities'].astype(str)

fig = plot_choropleth_df(locations_series, data_series, text_series)

###############################

drivers_per_state_2016_df = pd.read_csv('test.csv')

drivers_per_state_2016_df.head()

drivers_per_state_2016_df.reset_index(inplace=True)


drivers_per_state_2016_df.head()



#drivers_per_state_2016_df['index']+=1
drivers_per_state_2016_df['state_code'] = drivers_per_state_2016_df['State'].map(us_state_abbrev)
drivers_per_state_2016_df



drivers_per_state_2016_df.set_index('state_code', inplace=True)
drivers_per_state_2016_df = drivers_per_state_2016_df.iloc[:51]

accidents_state_df.head()
accidents_state_df['state_code'] = accidents_state_df['state_name'].map(us_state_abbrev)


accidents_state_df['licensed_drivers_k'] = accidents_state_df['state_code'].map(drivers_per_state_2016_df['Licensed Drivers (Thousands)'])


accidents_state_df['fatalities_per_100k_drivers'] = accidents_state_df['state_code'].map(drivers_per_state_2016_df['Fatalities per 100,000 Drivers'])


accidents_state_df.head()


#########################

QUERY = """SELECT
  state_number,
  state_name,
  COUNT(consecutive_number) AS accidents,
  SUM(number_of_fatalities) AS fatalities,
  SUM(number_of_fatalities) / COUNT(consecutive_number) AS fatalities_per_accident
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
GROUP BY
  state_number, state_name
ORDER BY
  state_number """
  
  
  
query_job=client.query(
QUERY.format(2015),
location='US',
job_config=bigquery.QueryJobConfig(maximum_bytes_billed=50_000_000))



#for row in query_job:
#    print(row)


state_number=[]
state_name=[]
accidents=[]
fatalities=[]
fatalities_per_accident=[]
for row in query_job.result():
    state_number.append(row[0])
    state_name.append(row[1])
    accidents.append(row[2])
    fatalities.append(row[3])
    fatalities_per_accident.append(row[4])
cols={'state_number':state_number, 'state_name':state_name, 'accidents':accidents, 'fatalities':fatalities, 'fatalities_per_accident':fatalities_per_accident}
    
cols


state_fatalities_rate = pd.DataFrame(cols)
state_fatalities_rate


plt.scatter(state_fatalities_rate.accidents,state_fatalities_rate.fatalities)
plt.title("Fatal Acidents vs Fatalities")
plt.xlabel("Fatal Accidents")
plt.ylabel("Fatalities")
plt.show()


##################


q_accidets_drunk = '''
SELECT
  state_number,
  count(number_of_drunk_drivers) AS drunk_drivers_count
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
WHERE number_of_drunk_drivers > 0
GROUP BY state_number
ORDER BY
  state_number
'''



query_job=client.query(
q_accidets_drunk.format(2016),
location='US',
job_config=bigquery.QueryJobConfig(maximum_bytes_billed=50_000_000))



#for row in query_job:
#    print(row)

a=[]
b=[]
for row in query_job:
    a.append(row[0])
    b.append(row[1])
drk = {'state_number': a, 'drunk_drivers_count': b}
accidets_drunk_df = pd.DataFrame(drk)


accidets_drunk_df.head() 



q_vehicle_pickup = '''
SELECT
  state_number,
  COUNT(body_type) AS pickups_count
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{0}`
WHERE body_type IN (30,31)
GROUP BY state_number
ORDER BY
  state_number
'''


query_job=client.query(
q_vehicle_pickup.format(2016),
location='US',
job_config=bigquery.QueryJobConfig(maximum_bytes_billed=50_000_000))



#for row in query_job:
#    print(row)


a=[]
b=[]
for row in query_job:
    a.append(row[0])
    b.append(row[1])
drk = {'state_number': a, 'pickups_count': b}
vehicle_pickup_df = pd.DataFrame(drk)

vehicle_pickup_df.head()



## Stacking it with the initial dataframe to get the count per 100 fatalities by state

accidents_state_df['drunk_drivers_count'] = accidets_drunk_df['drunk_drivers_count'] / accidents_state_df['fatalities']*100
accidents_state_df['pickups_count'] = vehicle_pickup_df['pickups_count'] / accidents_state_df['fatalities']*100
sample_state_accidents_df = accidents_state_df[accidents_state_df['state_name'].isin(['New York','Mississippi'])]
sample_state_accidents_df


def barplot_drunkards():
    # Define Variables
    fatalities_per_100k = sample_state_accidents_df['fatalities_per_100k_drivers']
    drunk_drivers = sample_state_accidents_df['drunk_drivers_count']
    ind = np.arange(2)
    width = 0.7

    # Plot graph
    gs = gridspec.GridSpec(1, 3)
    fig= plt.figure(figsize=(10,6));
    ax1 = fig.add_subplot(gs[:,:-1])
    ax2 = fig.add_subplot(gs[:,-1:])
    bars_drunk_drivers = ax1.bar(ind, drunk_drivers, width, linewidth = 0.3, color=sns.xkcd_rgb["purple"], edgecolor='black', alpha = 0.8, label='Test');
    bars_tot_fatal = ax2.bar(ind, fatalities_per_100k, width, linewidth = 0.3, edgecolor='black', alpha = 0.8, label='Fatality Rate');

    ax1.set_title('Drunks per 100 accidents\nMissisipi and NY', alpha=0.7, fontsize=14)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['Missisipi drunk drivers', 'New York drunk drivers'])
    ax1.set_ylim(0,35)
    ax1.set_ylabel('Drunkards per 100 fatalities', alpha = 0.7)
    ax1.axhline(15, lw=0.6, linestyle='--')

    ax2.set_title('Total Fatalities\nper 100k drivers', alpha=0.7, fontsize=14)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['Missisipi', 'New York']);
    
    
def barplot_pickups():
    # Define Variables
    fatalities_per_100k = sample_state_accidents_df['fatalities_per_100k_drivers']
    drunk_drivers = sample_state_accidents_df['pickups_count']
    ind = np.arange(2)
    width = 0.7

    # Plot graph
    gs = gridspec.GridSpec(1, 3)
    fig= plt.figure(figsize=(10,8));
    ax1 = fig.add_subplot(gs[:,:-1])
    ax2 = fig.add_subplot(gs[:,-1:])
    bars_drunk_drivers = ax1.bar(ind, drunk_drivers, width, linewidth = 0.3, color=sns.xkcd_rgb["purple"], edgecolor='black', alpha = 0.8, label='Test');
    bars_tot_fatal = ax2.bar(ind, fatalities_per_100k, width, linewidth = 0.3, edgecolor='black', alpha = 0.8, label='Fatality Rate');

    ax1.set_title('Pickups per 100 accidents\nMissisipi and NY', alpha=0.7, fontsize=14)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['Missisipi Pickup Cars', 'New York Pickup Cars'])
    ax1.set_ylim(0,35)
    ax1.set_ylabel('Pickups per 100 fatalities', alpha = 0.7)
    ax1.axhline(12.1, lw=0.6, linestyle='--')

    ax2.set_title('Total Fatalities\nper 100k drivers', alpha=0.7, fontsize=14)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['Missisipi', 'New York']);



barplot_drunkards()

barplot_pickups()

########################

q_vehicle = '''
SELECT
  state_number,
  vehicle_make_name,
  body_type_name,
  vehicle_model_year,
  travel_speed,
  compliance_with_license_restrictions,
  previous_recorded_suspensions_and_revocations,
  previous_speeding_convictions,
  speeding_related,
  trafficway_description,
  speed_limit,
  roadway_alignment,
  roadway_grade,
  roadway_surface_type,
  roadway_surface_condition
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{0}`
ORDER BY
  state_number
'''


query_job=client.query(
q_vehicle.format(2016),
location='US',
job_config=bigquery.QueryJobConfig(maximum_bytes_billed=50_000_000))


#for row in query_job:
#    print(row)

a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]
i=[]
j=[]
k=[]
l=[]
m=[]
n=[]
o=[]
for row in query_job:
    a.append(row[0])
    b.append(row[1])
    c.append(row[2])
    d.append(row[3])
    e.append(row[4])
    f.append(row[5])
    g.append(row[6])
    h.append(row[7])
    i.append(row[8])
    j.append(row[9])
    k.append(row[10])
    l.append(row[11])
    m.append(row[12])
    n.append(row[13])
    o.append(row[14])
vehicle_make = {'state_number': a, 'vehicle_make_name': b,'body_type_name': c,'vehicle_model_year': d,'travel_speed': e,'compliance_with_license_restrictions': f,'previous_recorded_suspensions_and_revocations': g,'previous_speeding_convictions': h,'speeding_related': i,'trafficway_description': j,'speed_limit': k,
      'roadway_alignment': l,'roadway_grade': m,'roadway_surface_type': n, 'roadway_surface_condition': o}
vehicle_df = pd.DataFrame(vehicle_make)


vehicle_df.head()