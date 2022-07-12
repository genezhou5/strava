# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:19:07 2022

@author: gzhou
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash()
app.layout = html.Div([
    html.H1(id='live-counter'),
    dcc.Interval(
        id='1-second-interval',
        interval=1000, 
        n_intervals=0
    )
])

@app.callback(Output('live-counter', 'children'),
              [Input('1-second-interval', 'n_intervals')])

def update_layout(n):
    return 'This app has updated itself for {} times, every second.'.format(n)

if __name__ == '__main__':
    app.run_server()