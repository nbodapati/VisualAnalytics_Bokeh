import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
from holoviews.streams import Stream,param

from sklearn import datasets as datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knn
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


data=datasets.load_iris(return_X_y=False)
X=data['data']
Y=data['target']
target_names=data['target_names']
feature_names=data['feature_names']
print(X.shape,Y.shape,target_names,feature_names)

X=pd.DataFrame(X,columns=['sepal_length','sepal_width','petal_length','petal_width'])
X['target']=Y
print(X.head())

'''
#this is of the type array[features,targets]
X=source.to_df()

'''


def heatmap_to_pair(viz_params=None,source=None):

    #0 -train, 1- test
    def train_test_split():
        global X
        X['train_test']=1

        class0=X[X['target']==0]
        class0=class0.reset_index(drop=True)
        N=int(class0.shape[0]/2)
        class0.loc[0:N-1,'train_test']=0

        class1=X[X['target']==1]
        class1=class1.reset_index(drop=True)
        N=int(class1.shape[0]/2)
        class1.loc[0:N-1,'train_test']=0

        class2=X[X['target']==2]
        class2=class2.reset_index(drop=True)
        N=int(class2.shape[0]/2)
        class2.loc[:N-1,'train_test']=0
    
        data=[class0,class1,class2]
        data=pd.concat(data).reset_index(drop=True) 

        train=data[data['train_test']==0]
        test=data[data['train_test']==1]
        data=[train,test] 
        data=pd.concat(data)
    
        train=np.array(train)
        x_train=train[:,:-2]
        y_train=train[:,-2]

        test=np.array(test)
        x_test=test[:,:-2]
        y_test=test[:,-2]

        return (x_train,y_train,x_test,y_test,data)


    def train_evaluate_model(model,X_train,y_train,X_test,y_test):
        if(model=='DT'):
          dt=DecisionTreeClassifier(criterion='entropy',
                         max_depth=5, min_samples_split=2, 
                         min_samples_leaf=1)
          dt.fit(X_train,y_train)
          train_pred=dt.predict(X_train)
          test_pred=dt.predict(X_test)
          acc=np.mean(test_pred==y_test)  
          test_viz=[]
          for i in range(X_test.shape[0]):
              test_viz.append((i,y_test[i],test_pred[i])) 
        
        elif(model=='SVM'):
           svc=SVC(C=0.1,kernel='rbf')
           svc.fit(X_train,y_train)
           train_pred=svc.predict(X_train)
           test_pred=svc.predict(X_test)
           acc=np.mean(test_pred==y_test)
           test_viz=[]
           for i in range(X_test.shape[0]):
               test_viz.append((i,y_test[i],test_pred[i]))         

        elif(model=='KNN'):
            knn_=knn(n_neighbors=1)
            knn_.fit(X_train,y_train)
            train_pred=knn_.predict(X_train)
            test_pred=knn_.predict(X_test)
            acc=np.mean(test_pred==y_test)
            test_viz=[]
            for i in range(X_test.shape[0]):
                test_viz.append((i,y_test[i],test_pred[i]))

        return train_pred,test_pred,test_viz,acc 

    models=['DT','SVM','KNN']
    x_train,y_train,x_test,y_test,data=train_test_split()
    data['index']=list(range(data.shape[0]))

    source=ColumnDataSource(data)

    PLOT_OPTS=dict(height=250,width=500,toolbar_location="above")
    colors = ["#75968f", "#e2e2e2", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=data.target.min(), high=data.target.max())
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    df=data
    columns = sorted(df.columns)
    for x in columns:
        print(x,df[x].dtype,len(set(df[x])))

    discrete = [x for x in columns if  len(set(df[x]))<7]
    categories = [x for x in discrete if df[x].dtype == object]
    continuous = [x for x in columns if x not in discrete]
    quantileable = [x for x in continuous if len(df[x].unique()) > 20]
    #print(columns,discrete,categories,continuous,quantileable)
    print("discrete: ",discrete)
    print("categories: ",categories)
    print("continuous: ",continuous)
    print("quantileable: ",quantileable)

    x_value=viz_params.get('x_value','index')
    y_value=viz_params.get('y_value','target')
    color_value=viz_params.get('color_value','pred')
    print(x_value,y_value,color_value)
    
    source.add(source.data[x_value],'x')
    source.add(source.data[y_value],'y')
      
    def bokeh_scatter():
        hover=HoverTool(tooltips=[\
                           ('target','@target')],show_arrow=False)
        p=figure(tools='pan,box_select,reset',**PLOT_OPTS)
        p.add_tools(hover)
        y_='sepal_width'
        p.circle(x='index',y='y',\
             source=source,\
             fill_color={'field':'target', 'transform': mapper})
        return p
                     
    def bokeh_heatmap():  
        train_pred,test_pred,_,_=train_evaluate_model(select_.value,x_train,y_train,x_test,y_test)
        try:
           print(source.data['pred'])
        except:
           data['pred']=list(train_pred)+list(test_pred)
           source.add(data['pred'],'pred')
           source.add(source.data['pred'],'color')  
              
        print(select_.value)
        title="Predictions: %s"%(select_.value)

        x_value=viz_params.get('x_value','index')
        y_value=viz_params.get('y_value','target')
        color_value=viz_params.get('color_value','pred')
        print(x_value,y_value,color_value)

        hover1=HoverTool(tooltips=[('pred_value',"@pred"),\
                           ('target','@target')],show_arrow=False)
    
        p1 = figure(tools='pan,box_select,reset',**PLOT_OPTS)
        p1.add_tools(hover1)
    
        p1.rect(x='x', y='y', width=1, height=1,
              fill_color={'field':'color', 'transform': mapper},
               source=source,
               line_color=None)

        p1.xaxis.axis_label=x_value
        p1.yaxis.axis_label=y_value
        p1.title.text=title

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d"),
                     label_standoff=10, border_line_color=None, location=(0, 0))

        p1.add_layout(color_bar,'right')
        return p1

    def get_new_pred(model):
        train_pred,test_pred,_,_=train_evaluate_model(model,x_train,y_train,x_test,y_test)
        return list(train_pred)+list(test_pred) 

    def select_change(attr,old,new):
        viz_params['x_value']=x.value
        viz_params['y_value']=y.value
        viz_params['color_value']=color.value

        pred=get_new_pred(select_.value)
        print(np.mean(source.data['pred']==pred))
        source.data['color']=pred


    select_=Select(title="Select Classifier",value="DT",options=["DT","SVM","KNN"])
    select_.on_change('value',select_change)
   
    def update_plot(attr,old,new):
        viz_params['x_value']=x.value
        viz_params['y_value']=y.value
        viz_params['color_value']=color.value

        source.data['x']=source.data[x.value]
        source.data['y']=source.data[y.value]
        source.data['color']=source.data[color.value]

     
    x=Select(title='X-Axis', value=viz_params.get('x','index'), options=quantileable)
    x.on_change('value',update_plot)

    y = Select(title='Y-Axis', value=viz_params.get('y','target'), options=quantileable)
    y.on_change('value',update_plot)

    color = Select(title='Color', value=viz_params.get('color','pred'), options=['pred'] + categories+quantileable)
    color.on_change('value',update_plot)

    scatter=bokeh_scatter()
    hmap=bokeh_heatmap()
    #plot=layout([row(hmap,s)])
    plot=column(scatter,hmap)
    widgets=column(widgetbox(select_,x,y,color))
    return (plot,widgets)



plot,widgets=heatmap_to_pair(viz_params={})
l=layout([[row(plot,widgets)]])
curdoc().add_root(l)