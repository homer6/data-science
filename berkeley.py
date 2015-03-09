

import pandas as pd
import numpy as np





df = pd.read_csv( 'Berkeley.csv' )

#df['admitted_rate'] = df.

males = df[ df.Gender == 'Male' ]



df['admitted_males'] = males.groupby( ['Dept'] ).sum()

#print males.groupby( ['Dept'] ).sum()

#print df.groupby( ['Admit', 'Gender', 'Dept'] ).sum()


print df
#print df.groupby( ['Gender'] ).describe()