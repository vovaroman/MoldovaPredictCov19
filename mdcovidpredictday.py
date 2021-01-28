import pandas as pandas
import matplotlib.pyplot as plt
import numpy
from datetime import datetime
from sklearn.metrics import r2_score
from datetime import timezone
from datetime import datetime

df = pandas.read_csv('./WHO-COVID-19-global-data.csv')

def datetime_to_utc(dt):
    return dt.replace(tzinfo=timezone.utc).timestamp()

def utc_to_datetime(ts):
    ts = int(ts)
    # print(ts)
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')


df = df.loc[df['Country'] == 'Republic of Moldova']


df['Date_reported'] = pandas.to_datetime(df.Date_reported)


df.sort_values('Date_reported')

x = df['Date_reported']

x = x.apply(lambda x: int(datetime_to_utc(x)))

x = x.to_numpy()

y = df['New_cases'].to_numpy()

# x, y = y, x

# print(y)

model = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, model(x)))

print(model(datetime_to_utc(datetime.strptime('2021-1-27', '%Y-%m-%d'))))