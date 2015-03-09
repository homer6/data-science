import numpy
import pandas
import statsmodels.api as sm

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


pandas.set_option('display.height', 1000)
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)


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
    df['is_adult'] = ( df.Age >= 18 ) & ( df.Age < 40 )
    df['is_older'] = ( df.Age >= 40 )

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



    #df['rich_female'] = ( df.Sex == "female" ) & ( df.Pclass < 3 ) & ( df.Fare > 8 )
    #df['rich_female_2'] = ( df.Sex == "female" ) & ( df.Pclass < 3 )
    #df['rich_female_3'] = ( df.Sex == "female" ) & ( df.Pclass < 3 ) & ( df.Age < 6 )
    #df['predictor_1'] = ( ( df.Sex == "female" ) & ( df.Pclass < 3 ) ) | ( ( df.Age < 6 ) & ( df.Pclass < 3 ) )


    #df['predictor_2'] = ( df.is_female * 100 ) + ( df.class_points * 20 ) + ( 99 - df.Age ) + ( df.Fare )
    #df['predictor_2'] = ( df.is_female * 300 ) - ( df.is_male * 300 )  + ( df.class_1 * 100 ) - ( df.class_3 * 100 ) + ( df.Fare / 5 ) + ( df['deck_b'] * 50 ) + ( df['deck_c'] * 50 ) + ( df['deck_d'] * 50 ) + ( df['deck_e'] * 50 ) - ( df['deck_nan'] * 75 )
    df['predictor_2'] = \
        ( df.is_female * 200 ) - ( df.is_male * 200 ) + ( df.is_child * 50 ) + ( df.class_1 * 70 ) - ( df.class_3 * 70 ) + \
        ( df.Fare / 5 ) + ( df['deck_b'] * 50 ) + ( df['deck_c'] * 50 ) + ( df['deck_d'] * 50 ) + ( df['deck_e'] * 50 ) - ( df['deck_nan'] * 25 ) + \
        ( df.embarked_c * 10 ) - ( df.embarked_s * 10 ) + \
        ( df.parch_gt_0 * 50 )

    df['predictor_3'] = \
        ( df.SibSp / df.SibSp.max() * -0.035322499 ) + \
        ( df.Parch / df.Parch.max() * 0.081629407 ) + \
        ( df.Fare / df.Fare.max() * 0.257306522 ) + \
        ( df.is_male * -0.543351381 ) + \
        ( df.is_female * 0.543351381 ) + \
        ( df.class_points / df.class_points.max() * 0.338481036 ) + \
        ( df.is_young_child * 0.147718696 ) + \
        ( df.is_child * 0.122238974 ) + \
        ( df.is_adult * -0.000559549 ) + \
        ( df.is_older * -0.009345779 ) + \
        ( df.class_1 * 0.285903768 ) + \
        ( df.class_2 * 0.093348572 ) + \
        ( df.class_3 * -0.322308357 ) + \
        ( df.deck_a * 0.022286954 ) + \
        ( df.deck_b * 0.175095034 ) + \
        ( df.deck_c * 0.114652115 ) + \
        ( df.deck_d * 0.150715644 ) + \
        ( df.deck_e * 0.145321443 ) + \
        ( df.deck_f * 0.057934947 ) + \
        ( df.deck_g * 0.016040183 ) + \
        ( df.deck_t * -0.026456469 ) + \
        ( df.deck_nan * -0.316911523 ) + \
        ( df.embarked_c * 0.168240431 ) + \
        ( df.embarked_q * 0.003650383 ) + \
        ( df.embarked_s * -0.155660273 ) + \
        ( df.parch_gt_0 * 0.1474075 ) + \
        ( df.parch_gt_1 * 0.056346089 ) + \
        ( df.parch_gt_2 * -0.031527886 )

    df['predictor_4'] = \
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


    #df['predictor_3'] = ( df.is_female * 200 ) + ( df.class_points * 20 )
    #df['predictor_4'] = ( 200 - df.Age )
    #df['predictor_5'] = ( df.Fare / 5 )
    #df['predictor_6'] = ( df['deck_b'] * 50 ) + ( df['deck_c'] * 50 ) + ( df['deck_d'] * 50 ) + ( df['deck_e'] * 50 )
    #df['predictor_7'] = ( df.is_female * 200 ) + ( df.class_points * 20 ) + ( df.Fare / 5 ) + ( df['deck_b'] * 50 ) + ( df['deck_c'] * 50 ) + ( df['deck_d'] * 50 ) + ( df['deck_e'] * 50 )

    #df['rich_female_points'] = df.class_points
    #df[ df.is_female ]['rich_female_points'] += 4



    #print df[ ["Survived", "rich_female_3" ] ].sum()
    #print df[ df.Age > 56 ].Age.describe()

    #print df.predictor_2.describe()
    #print df.predictor_3.describe()
    print df.predictor_4.describe()



    df['will_survive'] = df.predictor_4 > -0.1
    df['correct'] = df.will_survive == df.Survived


    #print df

    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

    print df.correct.describe()


    for passenger_index, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
      
        # Your code here:
        # For example, let's assume that if the passenger
        # is a male, then the passenger survived.
        #if passenger['Sex'] == 'male':
        #    predictions[passenger_id] = 1

    df_corr = df.corr()
    print df_corr

    df_corr.to_csv( "titanic_corr.csv" )

    df.to_csv( "titanic_output.csv" )


    #df.Pclass.hist().get_figure().savefig( 'images/titanic_pclass_hist.png' )
    #df.Age.hist().get_figure().savefig( 'images/titanic_age_hist.png' )
    #df.is_female.hist().get_figure().savefig( 'images/titanic_sex_hist.png' )


    #print df.Pclass.describe()

    #print df.deck.unique()

    #print df.Survived.describe()

    return predictions


print simple_heuristic( "titanic_data.csv" )