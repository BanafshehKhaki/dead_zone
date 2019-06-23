#Viz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re

from plotly.offline import plot
import plotly.graph_objs as go
import plotly.plotly as py


testdirectory = 'Results/test/'
DO_TestResults = pd.read_csv(testdirectory+'dissolved_oxygen/sonde1_2018-06-01-2018-07-01_GRU35_v1.csv')
DO_catTestResults = pd.read_csv(testdirectory+'DOcategory/sonde1_2018-06-01-2018-07-01_GRU35_v1.csv')
pH_TestResults = pd.read_csv(testdirectory+'ph/sonde1_2018-06-01-2018-07-01_GRU35_v1.csv')
pH_catTestResults = pd.read_csv(testdirectory+'/pHcategory/sonde1_2018-06-01-2018-07-01_GRU35_v1.csv')




# z = DO_TestResults.Score.values
# z = z.reshape(xshape,yshape)
# dc_1 = go.Surface( x=x,
#                    y=y,
#                    z = z,
#                    name = 'dissolved oxygen RELU Test epoch 35',
#                  )
# z = pH_TestResults.lr_R2.values
# z = z.reshape(xshape,yshape)
# dc_2 = go.Surface(x = x,
#                   y = y,
#                   z = z,
#                   name = 'pH Test', 
#                 )
# 
# z = DO_catTestResults.Accuracy.values
# z = z.reshape(xshape,yshape)
# dc_3 = go.Surface( x=x,
#                    y=y,
#                    z = z,
#                    name = 'DO category test accuracy',
#                  )
# z = pH_catTestResults.Accuracy.values
# z = z.reshape(xshape,yshape)
# dc_4 = go.Surface(x = x,
#                   y = y,
#                   z = z,
#                   name = 'pH category test accuracy', 
#                 )

# data = [dc_3]
# layout = go.Layout(
#     title='RNN GRU Results DO Category',
# 
#     
#     scene = go.Scene(
#     xaxis=dict(
#         title='Prediction Horizon',
#         titlefont=dict(
#            family='Courier New, monospace',
#            size=18,
#            color='#7f7f7f'
#         )
#    ),
#     yaxis=dict( title='Temporal depth',
#               titlefont=dict(
#            family='Courier New, monospace',
#            size=18,
#            color='#7f7f7f'
#               )
#         ),
#     zaxis = dict( title='R2-Score',
#               titlefont=dict(
#            family='Courier New, monospace',
#            size=18,
#            color='#7f7f7f'
#               )
#         ),
#     )    
# )
# 
# fig = go.Figure(data=data, layout=layout)
# plot(fig, filename='RNN_GRU_DO_category.html')


iLoop= DO_TestResults.TD.unique()
jLoop= DO_TestResults.PrH.unique()
x = jLoop
y =iLoop 
xshape = len(iLoop)
yshape = len(jLoop)
z = DO_TestResults.Score.values
z = z.reshape(xshape,yshape)
dc_1 = go.Heatmap( x=x,
                   y=y,
                   z = z,
                   name = 'dissolved oxygen RELU Test epoch 35',
                 )
                 
iLoop= pH_TestResults.TD.unique()
jLoop= pH_TestResults.PrH.unique()
x = jLoop
y =iLoop 
xshape = len(iLoop)
yshape = len(jLoop)
z = pH_TestResults.Score.values
z = z.reshape(xshape,yshape)
dc_2 = go.Heatmap(x = x,
                  y = y,
                  z = z,
                   name = 'ph RELU Test epoch 35',
                )
         

iLoop= DO_catTestResults.TD.unique()
jLoop= DO_catTestResults.PrH.unique()
x = jLoop
y =iLoop 
xshape = len(iLoop)
yshape = len(jLoop)       
z = DO_catTestResults.Score.values
z = z.reshape(xshape,yshape)
dc_3 = go.Heatmap( x=x,
                   y=y,
                   z = z,
                   name = 'DO category RELU Test epoch 35 accuracy',
                 )


iLoop= pH_catTestResults.TD.unique()
jLoop= pH_catTestResults.PrH.unique()
x = jLoop
y =iLoop 
xshape = len(iLoop)
yshape = len(jLoop)   
z = pH_catTestResults.Accuracy.values
z = z.reshape(xshape,yshape)
dc_4 = go.Heatmap(x = x,
                  y = y,
                  z = z,
                  name = 'pH category test accuracy', 
                )

data = [dc_1,dc_2,dc_3]
fig = go.Figure(data=data)
plot(data, filename='Results/all heatmap.html')

