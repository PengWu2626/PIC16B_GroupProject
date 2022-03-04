import pandas as pd
import plotly
import plotly.express as px
import json

def dogtime_plot(which_dog):
  """
  make bar charts of input dog breed
  Parameters
  ----------
  which_dog: string; dog breed

  Return 
  ----------
  bar charts
  """
  results = pd.read_csv('static/dogtime.csv')
  df = results[results['breed']==which_dog]
  fig = px.bar(df, x='characteristic', y='star', color="category")
  fig.update_layout(title_text=f"{which_dog} Plot", title_x=0.5)
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

  # image url for the dog
  dogpic = df['image_src'].unique()[0]
  return graphJSON, dogpic   