import pandas as pd
import numpy as np
import string

def get_dog_characteristics_map(dogdf):
  """
  This function will get the unstacked DogTime dataframe and return a 
  dictionary of dog characteristics mapper.

  Parameters
  ----------
  dogdf: panda dataframe;  data from prepared DogTime dataframe

  Return 
  ----------
  get_dog_characteristics_map(dogdf): dict; dog characteristics mapper

  ----------
  dog_characteristics_map info (total 26)
 {'a': 'breed',
  'b': 'Adapts Well To Apartment Living',
  'c': 'Affectionate With Family',
  'd': 'Amount Of Shedding',
  'e': 'Dog Friendly',
  'f': 'Drooling Potential',
  'g': 'Easy To Groom',
  'h': 'Easy To Train',
  'i': 'Energy Level',
  'j': 'Exercise Needs',
  'k': 'Friendly Toward Strangers',
  'l': 'General Health',
  'm': 'Good For Novice Owners',
  'n': 'Intelligence',
  'o': 'Intensity',
  'p': 'Kid-Friendly',
  'q': 'Potential For Mouthiness',
  'r': 'Potential For Playfulness',
  's': 'Potential For Weight Gain',
  't': 'Prey Drive',
  'u': 'Sensitivity Level',
  'v': 'Size',
  'w': 'Tendency To Bark Or Howl',
  'x': 'Tolerates Being Alone',
  'y': 'Tolerates Cold Weather',
  'z': 'Tolerates Hot Weather'}
  """
  # we have 26 characteristics
  return dict(zip(list(string.ascii_lowercase),list(dogdf.columns[1:])))

def prepare_recommendation_df(dogdf = pd.read_csv('static/dogtime.csv')):
  """
  This function will prepare and unstack the DogTime dataframe

  Parameters
  ----------
  dogdf: panda dataframe;  data from DogTime
         default to dogtime.csv

  Return 
  ----------
  dogdf: panda dataframe; unstacked input dataframe
  get_dog_characteristics_map(dogdf): dict; dog mapper from function get_dog_characteristics_map()
  """
  dogdf = dogdf[['breed','characteristic','star']]
  dogdf = dogdf.set_index(keys=['breed','characteristic'])
  # unstack to columns with breed + 26 characteristics
  dogdf = dogdf.unstack()
  dogdf.columns = dogdf.columns.droplevel(0)
  dogdf = dogdf.reset_index()
  dogdf = dogdf.rename_axis(index=(None), columns=None)

  # 3 nan values
  # at rows [163, 245, 317]
  # [Doxiepoo, Korean Jindo Dog, Puginese]
  # ['Drooling Potential', 'Prey Drive', 'Tendency To Bark Or Howl']
  # replace nan to 0
  dogdf = dogdf.fillna(0)
  # change values in 26 characteristic columns to integer
  dogdf[dogdf.columns[1:]] = dogdf[dogdf.columns[1:]].astype(int)

  return dogdf, get_dog_characteristics_map(dogdf)


def make_characteristics_map(slider_val_array):
  """
  This function will get the list of user-selected characteristics number and
  add the operations on it, then return a dictionary which keys are 26 letters,
  values are associated operation with user-selected number

  Parameters
  ----------
  slider_val_array: array; array of user-selected 26 characteristics number from 
                   findyourdog.html

  Return 
  ----------
  characteristics: dict; dictionary which keys are 26 letters,
                   values are associated operation with user-selected number 
  """
  # list of 26 lower letters
  letters_26= list(string.ascii_lowercase)
  # if it is 0 , then change to ">=0", else " == #"
  slider_val_list_with_operation = list(np.where(slider_val_array =="0", ">=0", np.char.add("==", slider_val_array)))
  # make kwargs map for function dog_recommendation
  characteristics = dict(zip(letters_26, slider_val_list_with_operation))
  return characteristics


def find_dog_recommendation(**characteristics):
  """
  This function will return a list of matching dog for recommendation;
  based on the user-selected value of 26 dog characteristics.
  The characteristics information are scraped from DogTime.

  Parameters
  ----------
  **characteristics: kwargs; check dog_map from function of get_dog_map()
                             every 26 letters is a key argument associated with a characteristic
                             ex: 'a' is associated with 'Adapts Well To Apartment Living'
                             so input a = "==1" means find
                             star of 'Adapts Well To Apartment Living' is equal to 1
  Return 
  ----------
  results: list; list of matching dog recommendations
  """

  # get prepared dog dataframe with the dog characteristics mapper
  df, dog_map = prepare_recommendation_df()

  # make default to all 26 characteristics >=0
  characteristic_dict = dict.fromkeys(list(df.columns[1:]), ">=0")

  # loop over dictionary characteristics and update input number
  for key, val in characteristics.items():
    characteristic_dict[dog_map[key]]= val 

 # the query string to evaluate 
  cmd =\
  """ 
  `Adapts Well To Apartment Living`{Adapts Well To Apartment Living}
  &`Affectionate With Family`{Affectionate With Family}
  &`Amount Of Shedding` {Amount Of Shedding}
  &`Dog Friendly`{Dog Friendly}
  &`Drooling Potential`{Drooling Potential}
  &`Easy To Groom` {Easy To Groom}
  &`Easy To Train` {Easy To Train}
  &`Energy Level` {Energy Level}
  &`Exercise Needs` {Exercise Needs}
  &`Friendly Toward Strangers` {Friendly Toward Strangers}
  &`General Health` {General Health}
  &`Good For Novice Owners` {Good For Novice Owners}
  &`Intelligence`{Intelligence}
  &`Intensity`{Intensity}
  &`Kid-Friendly`{Kid-Friendly}
  &`Potential For Mouthiness`{Potential For Mouthiness}
  &`Potential For Playfulness`{Potential For Playfulness}
  &`Potential For Weight Gain`{Potential For Weight Gain}
  &`Prey Drive`{Prey Drive}
  &`Sensitivity Level`{Sensitivity Level}
  &`Size`{Size}
  &`Tendency To Bark Or Howl`{Tendency To Bark Or Howl}
  &`Tolerates Being Alone`{Tolerates Being Alone}
  &`Tolerates Cold Weather`{Tolerates Cold Weather}
  &`Tolerates Hot Weather`{Tolerates Hot Weather}
  """.format(**characteristic_dict).replace('\n','')

  # find matching dogs
  results = df.query(f'{cmd}')['breed'].tolist()
  return results
