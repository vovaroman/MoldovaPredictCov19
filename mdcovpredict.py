import pandas as pandas
import matplotlib.pyplot as plt
import numpy
from datetime import datetime
from sklearn.metrics import r2_score
from datetime import timezone
from datetime import datetime

df = pandas.read_csv('./covid-19-all.csv/covid-19-all.csv')

def display_data_for_country(df, country_name):
    
    df = df.loc[df['Country/Region'] == country_name]

    df['Date'] = pandas.to_datetime(df.Date)

    # print(df.Longitude.unique())

    # print(df.Latitude.unique())

    BBox = ((df.Longitude.min(),   df.Longitude.max(),      
         df.Latitude.min(), df.Latitude.max()))

    print(BBox)

    df.sort_values('Date')

    ax = df.plot(x='Date', y='Deaths', title = f'{country_name} Cov-19', linestyle='solid')

    ax = df.plot(x='Date', y='Recovered', ax=ax, linestyle='solid')

    ax = df.plot(x='Date', y='Confirmed', ax=ax, linestyle='solid')

def datetime_to_utc(dt):
    return dt.replace(tzinfo=timezone.utc).timestamp()

def utc_to_datetime(ts):
    ts = int(ts)
    # print(ts)
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')


def predict_cov19_confirmed(df, country_name, num):

    df = df.loc[df['Country/Region'] == country_name]
    
    df['Date'] = pandas.to_datetime(df.Date)
    
    df.sort_values('Date')

    x = df['Date']

    x = x.apply(lambda x: int(datetime_to_utc(x)))

    y = df['Confirmed'].to_numpy()

    x, y = y, x

    model = numpy.poly1d(numpy.polyfit(x, y, 3))

    print(r2_score(y, model(x)))

    return utc_to_datetime(model(num))





# display_data_for_country(df, 'Moldova')

print(predict_cov19_confirmed(df, 'Moldova', 157000))


