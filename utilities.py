'''
Module containing small general helper functions
'''

def showUniques(df):
  """
  Helper function that shows unique entries per DataFrame column.

  Arguments:
  df: the pandas DataFrame in question.
  """
  
  import pandas as pd
  from pandas import DataFrame 

  print("Number of rows: ", len(df))
  print("Number of unique values per column: ")
  for col in df.columns:
    print("Column {}: ".format(col), df[col].nunique())

#====================================
#====================================


def ensure_dir(directory):
  """When directory is not present, create it.

  Arguments: 
  directory: name of directory.
  """
  import os
  if not os.path.exists(directory):
    os.makedirs(directory)

#====================================
#====================================

def printBumper(text, c='=', n=-1):
  """Print a text with separating character. 

  Arguments: 
  text: text to be printed between separating character.

  c: separating character. 

  n: number of printed characters prior to the text (default -1: print as many characters as length of text)."""
  if n==-1:
    times = len(text)
  else:
    times = n
  print(c*times)
  print(text, ( (times-len(text)-1)*c ))
  print(c*times)

#====================================
#====================================

def intersec(d1, d2):
  """
  Returns list of intersecting elements.

  Arguments:
  d1/d2 - Objects that can be 'casted' as a sets.
  """
  return list(set(d1).intersection(set(d2)))

def union(d1,d2):
  """
  Returns list of union elements.

  Arguments:
  d1/d2 - Objects that can be 'casted' as a sets.
  """
  return list(set(d1).union(set(d2)))