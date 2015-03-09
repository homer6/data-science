import numpy
import pandas
import statsmodels.api as sm

def simple_heuristic(file_path):
    '''
    In this exercise, we will perform some rudimentary practices similar to those of
    an actual data scientist.
    
    Part of a data scientist's job is to use her or his intuition and insight to
    write algorithms and heuristics. A data scientist also creates mathematical models 
    to make predictions based on some attributes from the data that they are examining.

    We would like for you to take your knowledge and intuition about the Titanic
    and its passengers' attributes to predict whether or not the passengers survived
    or perished. You can read more about the Titanic and specifics about this dataset at:
    http://en.wikipedia.org/wiki/RMS_Titanic
    http://www.kaggle.com/c/titanic-gettingStarted
        
    In this exercise and the following ones, you are given a list of Titantic passengers
    and their associated information. More information about the data can be seen at the 
    link below:
    http://www.kaggle.com/c/titanic-gettingStarted/data. 

    For this exercise, you need to write a simple heuristic that will use
    the passengers' gender to predict if that person survived the Titanic disaster.
    
    You prediction should be 78% accurate or higher.
        
    Here's a simple heuristic to start off:
       1) If the passenger is female, your heuristic should assume that the
       passenger survived.
       2) If the passenger is male, you heuristic should
       assume that the passenger did not survive.
    
    You can access the gender of a passenger via passenger['Sex'].
    If the passenger is male, passenger['Sex'] will return a string "male".
    If the passenger is female, passenger['Sex'] will return a string "female".

    Write your prediction back into the "predictions" dictionary. The
    key of the dictionary should be the passenger's id (which can be accessed
    via passenger["PassengerId"]) and the associated value should be 1 if the
    passenger survied or 0 otherwise.

    For example, if a passenger is predicted to have survived:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 1

    And if a passenger is predicted to have perished in the disaster:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 0
    
    You can also look at the Titantic data that you will be working with
    at the link below:
    https://www.dropbox.com/s/r5f9aos8p9ri9sa/titanic_data.csv
    '''


    print "------------------------------------"

    predictions = {}
    df = pandas.read_csv(file_path)

    df['is_male'] = df.Sex == "male"
    df['is_female'] = df.Sex == "female"
    df['deck'] = df.Cabin.str[0]
    df['class_points'] = 4 - df.Pclass

    df['is_young_child'] = ( df.Age < 9 )
    df['is_child'] = ( df.Age < 18 )

    df['class_1'] = df.Pclass == 1
    df['class_2'] = df.Pclass == 2
    df['class_3'] = df.Pclass == 3


    df['deck_a'] = ( df.deck == 'A' )
    df['deck_b'] = ( df.deck == 'B' )
    df['deck_c'] = ( df.deck == 'C' )
    df['deck_d'] = ( df.deck == 'D' )
    df['deck_e'] = ( df.deck == 'E' )
    df['deck_f'] = ( df.deck == 'F' )
    df['deck_g'] = ( df.deck == 'G' )
    df['deck_t'] = ( df.deck == 'T' )
    df['deck_nan'] = df.deck.isnull()


    df['embarked_c'] = ( df.Embarked == 'C' )
    df['embarked_q'] = ( df.Embarked == 'Q' )
    df['embarked_s'] = ( df.Embarked == 'S' )

    df['parch_gt_0'] = ( df.Parch > 0 )
    df['parch_gt_1'] = ( df.Parch > 1 )
    df['parch_gt_2'] = ( df.Parch > 2 )

    df['predictor_1'] = \
        ( df.is_male * -0.543351381 ) + \
        ( df.is_female * 0.543351381 ) + \
        ( df.is_young_child * 0.147718696 ) + \
        ( df.is_child * 0.122238974 ) + \
        ( df.class_1 * 0.285903768 ) + \
        ( df.class_3 * -0.322308357 ) + \
        ( df.deck_b * 0.175095034 ) + \
        ( df.deck_c * 0.114652115 ) + \
        ( df.deck_d * 0.150715644 ) + \
        ( df.deck_e * 0.145321443 ) + \
        ( df.deck_nan * -0.316911523 ) + \
        ( df.embarked_c * 0.168240431 ) + \
        ( df.embarked_s * -0.155660273 ) + \
        ( df.parch_gt_0 * 0.1474075 )

    df['will_survive'] = df.predictor_1 > -0.1
    df['correct'] = df.will_survive == df.Survived

    print df.correct.describe()

    for passenger_index, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
        predictions[ passenger_id ] = passenger['will_survive']

    return predictions


print simple_heuristic( "titanic_data.csv" )