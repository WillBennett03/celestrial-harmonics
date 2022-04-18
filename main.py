import numpy as np
from jplephem.spk import SPK
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
start = pd.to_datetime('1900-01-01')
np.set_printoptions(precision=3)


def conv_dates(x):
    return x.to_julian_date()

def distance(x): #as center is 0,0,0 no need to -0 from each element
    return (x**2).sum(axis=1)**0.5

class Time_series(): #will contrain all objects and arrays needed for analysis
    def __init__(self, start_date=None, end_date=None):
        self.start_date = start_date
        self.end_date = end_date
        self.dates = None
        self.kernel = SPK.open('data/de421.bsp')

        self.get_dow()
        self.get_sun_pos()
        self.abs_distance()
        self.create_df()

    def get_dow(self): #gets Dow index day data from the csv, only using close
        self.dow_df = pd.read_csv('data/DJ_destro.csv').filter(['Date', 'Close'])
        if isinstance(self.start_date, type(None)) and isinstance(self.end_date, type(None)):
            pass
        else:
            mask = (self.dow_df['Date'] > self.start_date) & (self.dow_df['Date'] <= self.end_date)
            self.dow_df = self.dow_df.loc[mask]
        self.dates = pd.to_datetime(self.dow_df['Date'])
        self.julian_dates = self.dates.apply(conv_dates).values

    def get_sun_pos(self): #uses SPK and de421 dataset to get sun position relative to solar systems center.
        self.pos = self.kernel[0,10].compute(self.julian_dates).T
        

    def abs_distance(self):
        self.distances = distance(self.pos)

    def create_df(self):
        self.df = self.dow_df
        self.df['Sun'] = self.distances
        print(self.df.head())

    def graph_ts(self):
        scaler = MinMaxScaler(feature_range=(-1,1))
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(self.dates, scaler.fit_transform(self.df['Close'].values.reshape((-1,1))), label='Close')
        ax[0].plot(self.dates, scaler.fit_transform(np.log(self.df['Close'].values).reshape((-1,1))), label='Log(close)')
        ax[0].plot(self.dates, scaler.fit_transform(self.df['Sun'].values.reshape((-1,1))), label='Sun distance')
        ax[1].plot(self.dates, scaler.fit_transform(self.df['Close'].pct_change().values.reshape((-1,1))), label='Close')
        ax[1].plot(self.dates, scaler.fit_transform(np.log(self.df['Close']).pct_change().values.reshape((-1,1))), label='Log(Close)')
        ax[1].plot(self.dates, scaler.fit_transform(self.df['Sun'].pct_change().values.reshape((-1,1))), label='Sun distance')
        plt.legend(ax, ['Close', 'Log', 'Sun'])
        print(self.df.corr())
        plt.show()

    def graph_sun(pos):
        pass

if __name__ == '__main__':
    test = Time_series('1970-01-01', '2022-01-01')
    test.graph_ts()