#This code uses bokeh library
#and streams in data from data points.
from bokeh.events import ButtonClick
from bokeh.models import Button

import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file,show
import warnings
warnings.filterwarnings('ignore')

from bokeh.models import ColumnDataSource
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models.widgets import Dropdown,Select,TextInput
from bokeh.io import curdoc
from bokeh.models import (
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)
from bokeh.palettes import inferno
from bokeh.models.ranges  import DataRange1d

table=pd.read_table('./TSOCS-failures-finalV.txt',header=None,
                    delimiter=',')
table.columns=["col"+str(x) for x in range(table.shape[1])]
table['index']=range(table.shape[0])
source=ColumnDataSource(table)
source.add(source.data['col0'],'y')
source.column_names

source2=ColumnDataSource(dict(x=[],y=[]))
select_col=None
num_pts=10000
scatter=None

class Source_generator():
    def __init__(self,num_pts=10000):
        self.dataframe=table
        self.N=table.shape[0]
        self.num_pts=num_pts
        self.pts=0

    def __next__(self):
        #send next 500 points each time.
        #implement as a generator.
        global table, num_pts
        while(self.pts<self.N):
            pts,self.pts=self.pts,self.pts+self.num_pts
            print(pts)
            return table.loc[pts:pts+num_pts-1,:]

gen_objs={}
for i in source.column_names[:-2]:
    gen_objs[i]=Source_generator()

def update_plot():
    global select_col,num_pts,source2,scatter,gen
    col=select_col.value
    new_src = next(gen_objs[col])
    scatter.title.text= "Column values: %s Num Points %d"%(select_col.value,\
                             len(source2.data['x']))

    source2.stream(dict(x=new_src['index'],\
                     y=new_src[col]),100000)


PLOT_OPTS=dict(height=650,width=900,toolbar_location="above")
colors = inferno(256)
mapper = LinearColorMapper(palette=colors, low=min(table['col0']), high=max(table['col0']))
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
hover=HoverTool(tooltips=[('col_value','@y')],show_arrow=False)
p=figure(tools='pan,box_select,reset',**PLOT_OPTS)
p.add_tools(hover)
p.circle(x='x', y='y', source=source2, size=5,\
             fill_color={'field':'y', 'transform': mapper})

p.xaxis.axis_label='Datapoints'
p.yaxis.axis_label='col0'
scatter=p

button = Button()
def callback_button(event):
    print('Python:Click')
    update_plot()

button.on_event(ButtonClick, callback_button)
def select_change(attr,old,new):
    global scatter,source2,select_col
    scatter.title.text= "Column values: %s Num Points %d"%(select_col.value,0)
    col=select_col.value
    p.yaxis.axis_label=col
    source2.data['x']=[]
    source2.data['y']=[]

select_col=Select(title="Select Column",value="col0",options=source.column_names[:-2])
select_col.on_change('value',select_change)
plot=row(p,column(widgetbox(button),widgetbox(select_col)))
curdoc().add_root(plot)
