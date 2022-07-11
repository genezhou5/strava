# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:57:37 2022

@author: gzhou

Strava Stuff
"""

# Set Up
from os import chdir
import numpy as np
import pandas as pd
import math
import datetime
from plotly import express as px
from plotly import graph_objects as go
import plotly.io as pio # to render plots in browser

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output


chdir('C:\\Users\\gzhou\\Documents\\Strava\\Data')
pio.renderers.default='browser'

app = dash.Dash()
app = dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#############
# Functions #
#############

def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return(math.sqrt(variance))

# Data prep
activities_import = pd.read_csv('activities.csv')
activities = activities_import.rename(columns = {
    'Activity ID':'activity_id',
    'Activity Date':'activity_date_str',
    'Activity Name':'activity_name',
    'Activity Type':'activity_type',
    'Activity Description':'activity_description',
    'Elapsed Time':'elapsed_time_s',
    'Distance':'distance_km',
    'Max Heart Rate':'max_heart_rate',
    'Relative Effort':'relative_effort',
    'Commute':'commute',
    'Activity Gear':'activity_gear',
    'Filename':'filename',
    'Athlete Weight':'athlete_weight',
    'Bike Weight':'bike_weight',
    'Elapsed Time.1':'elapsed_time_1',
    'Moving Time':'moving_time',
    'Distance.1':'distance_m',
    'Max Speed':'max_speed',
    'Average Speed':'average_speed',
    'Elevation Gain':'elevation_gain',
    'Elevation Loss':'elevation_loss',
    'Elevation Low':'elevation_low',
    'Elevation High':'elevation_high',
    'Max Grade':'max_grade',
    'Average Grade':'average_grade',
    'Average Positive Grade':'average_positive_grade',
    'Average Negative Grade':'average_negative_grade',
    'Max Cadence':'max_cadence',
    'Average Cadence':'average_cadence',
    'Max Heart Rate.1':'max_heart_rate_1',
    'Average Heart Rate':'average_heart_rate',
    'Max Watts':'max_watts',
    'Average Watts':'average_watts',
    'Calories':'calories',
    'Max Temperature':'max_temperature',
    'Average Temperature':'average_temperature',
    'Relative Effort.1':'relative_effort_1',
    'Total Work':'total_work',
    'Number of Runs':'number_of_runs',
    'Uphill Time':'uphill_time',
    'Downhill Time':'downhill_time',
    'Other Time':'other_time',
    'Perceived Exertion':'perceived_exertion'
    }
    )
# remove distance_km bc it's not a number somehow
activities = activities.drop(['distance_km'], axis=1)

activities['activity_date'] = pd.to_datetime(activities['activity_date_str'],
                                             format='%b %d, %Y, %I:%M:%S %p')

activities['activity_date'] = activities['activity_date'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

activities['activity_week'] = activities['activity_date'] - activities['activity_date'].dt.weekday * np.timedelta64(1, 'D')
activities['activity_week'] = pd.to_datetime(activities['activity_week']).dt.date
activities['elapsed_time_min'] = activities['elapsed_time_1']/60
activities['distance_mi'] = activities['distance_m'] * 0.00062137
activities['distance_km'] = activities['distance_m']/1000

#df['First_day'] = df['Date'] - df['Date'].dt.weekday * np.timedelta64(1, 'D')

activities = activities[
    ['activity_id',
     'activity_date_str',
     'activity_date',
     'activity_week',
     'activity_name',
     'activity_type',
     'elapsed_time_s',
     'elapsed_time_min',
     'distance_m',
     'distance_km',
     'distance_mi',
     'elevation_gain',
     'elevation_loss',
     'elevation_low',
     'elevation_high',
     'max_grade',
     'average_grade',
     'average_positive_grade',
     'average_negative_grade',
     'uphill_time',
     'downhill_time',
     'other_time',
     'perceived_exertion'
     ]
    ]

runs = activities[(activities['activity_type']=='Run')
                  & (activities['activity_date'].dt.date > datetime.date(year=2022,month=2,day=20))]

# "A value is trying to be set on a copy of a slice from a DataFrame..."
#runs['pace'] = runs['elapsed_time_min']/runs['distance_mi']

runs = runs.assign(pace = runs['elapsed_time_min']/runs['distance_mi'])

runs_gb = runs[['activity_week',
                    'distance_km',
                    'distance_mi',
                    'elapsed_time_min',
                    'pace',
                    'perceived_exertion']].groupby(['activity_week']) #.agg(['sum', 'mean']).reset_index()

runs_weekly1 = runs_gb.agg(['sum', 'mean']).reset_index()

runs_weekly1.columns = ['activity_week',
                        'sum_distance_km',
                        'avg_distance_km',
                        'sum_distance_mi',
                        'avg_distance_mi',
                        'sum_elapsed_time_min',
                        'avg_elapsed_time_min',
                        'sum_pace',
                        'avg_pace',
                        'sum_perceived_exertion',
                        'avg_perceived_exertion']

runs_weekly1 = runs_weekly1[['activity_week',
                            #'sum_distance_km',
                            #'avg_distance_km',
                            'sum_distance_mi',
                            'avg_distance_mi',
                            'sum_elapsed_time_min',
                            'avg_elapsed_time_min',
                            'avg_perceived_exertion']]

runs_wt = pd.concat(
    [runs_gb.apply(lambda x: np.average(x['pace'],weights=x['distance_mi'])),
     runs_gb.apply(lambda x: weighted_std(values=x['pace'],weights=x['distance_mi']))],
    axis=1).reset_index()
runs_wt.columns = ['activity_week', 'avg_pace_wt', 'std_pace_wt']

runs_weekly = pd.merge(runs_weekly1, runs_wt, how='left', on=['activity_week'])

runs_mi_plt = px.line(runs_weekly, x='activity_week', y='sum_distance_mi')
runs_mi_plt.update_layout(
    title={
        'text': "Weekly Mileage",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title='Mileage')

runs_pace_plt = go.Figure()
runs_pace_plt.add_trace(go.Scatter(x=runs_weekly['activity_week'],
                              y=runs_weekly['avg_pace_wt'],
                              line=dict(color='rgba(51,51,255,1)'),
                              showlegend=False))
runs_pace_plt.add_trace(go.Scatter(x=runs_weekly['activity_week'],
                              y=runs_weekly['avg_pace_wt']+runs_weekly['std_pace_wt'],
                              line=dict(color='rgba(153,204,255,0)'),
                              fill='tonexty',
                              showlegend=False))
runs_pace_plt.add_trace(go.Scatter(x=runs_weekly['activity_week'],
                              y=runs_weekly['avg_pace_wt']-runs_weekly['std_pace_wt'],
                              line=dict(color='rgba(153,204,255,0)'),
                              fill='tonexty',
                              showlegend=False))
runs_pace_plt.update_layout(
    title={
        'text': "Avg Pace (min/mi)",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title='Pace')

swims = activities[(activities['activity_type']=='Swim')
                   & (activities['activity_date'].dt.date > datetime.date(year=2022,month=2,day=20))]

swims_weekly = swims[['activity_week',
                      'distance_m',
                      'elapsed_time_min',
                      'perceived_exertion']].groupby(['activity_week']).agg(['sum', 'mean']).reset_index()

swims_weekly.columns = ['activity_week',
                        'sum_distance_m',
                        'avg_distance_m',
                        'sum_time_min',
                        'avg_time_min',
                        'sum_exertion',
                        'avg_exertion']

swims_dist_plt = px.line(swims_weekly, x='activity_week', y='sum_distance_m')
swims_dist_plt.update_layout(
    title={
        'text':'Weekly Swim Distance (m)',
        'y':0.95,
        'x':0.5,
        'xanchor':'center',
        'yanchor': 'top'},
    xaxis_title='Date',
    yaxis_title='Meters Swum')

"""
app.layout = html.Div(children=[
    dcc.Graph(id='runs_mi_plt', figure=runs_mi_plt),
    dcc.Graph(id='runs_pace_plt', figure=runs_pace_plt),
    dcc.Graph(id='swims_dist_plt', figure=swims_dist_plt)
    ])
"""

app = dash.Dash()
# `Dash` doesn't work
#app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div(children=[dcc.Graph(id='runs_mi_plt',
                                             figure=runs_mi_plt)]),
                width='auto'),
        dbc.Col(html.Div(children=[dcc.Graph(id='runs_pace_plt',
                                             figure=runs_pace_plt)]),
                width='auto')
        ]),
    dbc.Row([
        dbc.Col(html.Div(children=[dcc.Graph(id='swims_dist_plt',
                                             figure=swims_dist_plt)]),
                width='auto')
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

#@app.callback(Output('graphid', 'figure'))

############################
# TEST CODE PLEASE IGNORE
############################

tapi = 'Sep 03, 2021, 12:13:10 AM'
tapi_dt = pd.to_datetime(tapi, format='%b %d, %Y, %I:%M:%S %p')
tapi_weekday = tapi_dt.weekday * np.timedelta64(1, 'D')


