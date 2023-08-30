import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA,NBEATSModel,Prophet,StatsForecastAutoARIMA, RandomForest
import statsmodels.api as sm
from pylab import rcParams

def test_stationarity(timeseries):
    plt.figure(figsize=(50, 8))
    rolmean = timeseries.rolling(window=3).mean()
    rolstd = timeseries.rolling(window=3).std()

    plt.plot(timeseries, color='blue', label='Originale')
    plt.plot(rolmean, color='red', label='Moyenne mobile')
    plt.plot(rolstd, color='black', label='Ecart-type mobile')
    plt.legend(loc='best')
    plt.title('Moyenne et Ecart-type mobile')
    plt.show(block=True)

def decomposition(timeseries):
    decomposition = sm.tsa.seasonal_decompose(timeseries, model='multiplicative')

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    rcParams['figure.figsize'] = 25, 10

    plt.subplot(411)
    plt.title('Observé = Tendance + Saisonnalité + Résidus')
    plt.plot(timeseries, label='Observé')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Tendance')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Saisonnalité')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Résidus')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

train = pd.read_csv("data/train_1.csv")
train_flattened = pd.melt(train[list(train.columns[-50:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

train_flattened_mean = train_flattened.groupby(['date'])['Visits'].mean().reset_index()
Train_flattened_index = train_flattened_mean.set_index('date', inplace=False)

#test_stationarity(Train_flattened_index['Visits'])
#decomposition(Train_flattened_index['Visits'])


time_series = TimeSeries.from_dataframe(train_flattened_mean, 'date', 'Visits')
train_data, val_data=time_series[:-8], time_series[-8:]

##############################################################
#ARIMA model 

model_data = ARIMA(p=7, d=1, q=2)
model_data.fit(train_data)
pred = model_data.predict(42)

##############################################################
#NBEATS Model

#model_data =NBEATSModel(input_chunk_length=16, output_chunk_length=8, n_epochs=100)
#model_data.fit([time_series],verbose=True)
#pred= model_data.predict(n=42, series=train_data)

##############################################################
#Prophet model 

#model_data = Prophet()
#model_data.fit(train_data)
#pred = model_data.predict(42)

##############################################################
#StatsForecastAutoARIMA model 

#model_data = StatsForecastAutoARIMA(season_length=7)
#model_data.fit(train_data)
#pred = model_data.predict(42)

##############################################################
#RandomForestmodel 

#model_data = RandomForest(lags=8)
#model_data.fit(train_data)
#pred = model_data.predict(42)


time_series.plot(label='réel')
pred.plot(label='prediction')
plt.legend()
plt.show()
