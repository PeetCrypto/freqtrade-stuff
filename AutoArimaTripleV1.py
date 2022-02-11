# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import os

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, informative)
from freqtrade.data.dataprovider import DataProvider

from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from pmdarima.arima import auto_arima

from datetime import datetime

pd.set_option('display.max_columns', None)


class AutoArimaTripleV1(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    log = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../logs/strategy-log.log'), 'w')

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    predictor_30m = None
    predictor_1h = None
    predictor_5m = None
    last_predicted_frame_30m = None, None
    last_predicted_frame_1h = None, None
    last_predicted_frame_5m = None, None

    @informative('30m')
    def populate_indicators_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_date = dataframe.loc[len(dataframe) - 1].date
        last_date, last_prediction = self.last_predicted_frame_30m
        if last_date == current_date:
            self.__print('Use stored predicted value as 30m indicator.')
            dataframe['prediction'] = dataframe.apply(lambda row: last_prediction if row.date == current_date else None, axis=1)
            return dataframe

        if not self.predictor_30m:
            self.predictor_30m = ArimaPredictor(self.dp, metadata['pair'], '30m', '30min')

        self.__print('===30m===')
        self.__print(dataframe.tail(2))
        prediction = self.predictor_30m.predict(dataframe, current_date)
        dataframe['prediction'] = dataframe.apply(lambda row: prediction if row.date == current_date else None, axis=1)
        self.last_predicted_frame_30m = current_date, prediction
        self.__print(f'Prediction 30m {current_date}: {prediction:0.2f}')

        # dry run: Remember last candle
        # backtest: process everything at the same time from head to tail
        # TODO Model generation in background
        # TODO Save and restore
        return dataframe

    def __print(self, message):
        self.log.write(f'{message}\n')

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_date = dataframe.loc[len(dataframe) - 1].date
        last_date, last_prediction = self.last_predicted_frame_1h
        if last_date == current_date:
            self.__print('Use stored predicted value as 1h indicator.')
            dataframe['prediction'] = dataframe.apply(lambda row: last_prediction if row.date == current_date else None, axis=1)
            return dataframe

        if not self.predictor_1h:
            self.predictor_1h = ArimaPredictor(self.dp, metadata['pair'], '1h', 'H')

        self.__print('===1h===')
        self.__print(dataframe.tail(2))
        prediction = self.predictor_1h.predict(dataframe, current_date)
        dataframe['prediction'] = dataframe.apply(lambda row: prediction if row.date == current_date else None, axis=1)
        self.last_predicted_frame_1h = current_date, prediction
        self.__print(f'Prediction 1h {current_date}: {prediction:0.2f}')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        current_date = dataframe.loc[len(dataframe) - 1].date
        last_date, last_prediction = self.last_predicted_frame_5m
        if last_date == current_date:
            self.__print('Use stored predicted value as 5m indicator.')
            dataframe['prediction'] = dataframe.apply(lambda row: last_prediction if row.date == current_date else None, axis=1)
            return dataframe

        if not self.predictor_5m:
            self.predictor_5m = ArimaPredictor(self.dp, metadata['pair'], '5m', '5min')

        self.__print('===5m===')
        self.__print(dataframe.tail(2))
        prediction = self.predictor_5m.predict(dataframe.loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']], current_date)
        dataframe['prediction'] = dataframe.apply(lambda row: prediction if row.date == current_date else None, axis=1)
        self.last_predicted_frame_5m = current_date, prediction
        self.__print(f'Prediction 5m {current_date}: {prediction:0.2f}')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        self.__print('===buy===')
        self.__print(dataframe.tail(1))
        dataframe.loc[
            (
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        self.__print('===sell===')
        self.__print(dataframe.tail(1))
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe


class ArimaPredictor(object):
    train_size = 10000
    steps = -1
    scale = None
    model: SARIMAXResults = None
    last_x = None

    def __init__(self, dp: DataProvider, pair: str, t_frame: str, frequency: str):
        self.dp = dp
        self.pair = pair
        self.t_frame = t_frame
        self.frequency = frequency
        self.data = self.__download_data()

    def predict(self, c_data: DataFrame, current_prediction_date):
        self.__combine_data(c_data)
        must_train = self.__must_train()
        # must be executed before get_current_data because the scale function is needed
        if must_train:
            self.__train(current_prediction_date)
        x, y = self.__get_current_data(current_prediction_date)
        if not must_train:
            self.model.append(y, exog=self.last_x, refit=False)
        prediction = self.model.forecast(steps=self.steps*-1, exog=x)
        self.last_x = x

        # TODO Confidence interval used in trend (only if lower and upper is in interval)
        prediction = prediction.tail(1).mean()
        current = x['mean'].mean()
        return (prediction-current)/current*100

    def __get_current_data(self, current_prediction_date):
        index = self.data.index.get_loc(current_prediction_date)
        row = self.data.iloc[index:index+1, :].copy().asfreq(freq=self.frequency)
        row['mean'] = (row['low'] + row['high']) / 2
        self.scale(row)
        return row, row.rename(columns={'mean': 'step_ahead'})\
                       .drop('open', axis=1)\
                       .drop('high', axis=1)\
                       .drop('low', axis=1)\
                       .drop('volume', axis=1)\
                       .drop('close', axis=1)

    def __must_train(self):
        return self.model is None

    def __train(self, current_prediction_date):
        start = datetime.now()
        print(f'Started {self.t_frame} training at {start}.')
        end_index = self.data.index.get_loc(current_prediction_date)
        x, y = self.__prepare_data(self.data, self.frequency)
        train_x, train_y, self.scale = self.__get_train_data(x, y, end_index)
        step_wise = auto_arima(train_y,
                               exogenous=train_x,
                               start_p=1, start_q=1,
                               max_p=7, max_q=7,
                               d=1, max_d=7,
                               trace=False,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
        print(f'Optimal parameter for {self.t_frame} are {step_wise.order}.')
        self.model = SARIMAX(train_y,
                             exog=train_x,
                             order=step_wise.order,
                             enforce_invertibility=False,
                             enforce_stationarity=False).fit()
        print(f'Finished training of {self.t_frame} in {datetime.now()-start}.')

    def __download_data(self):
        h_data = self.dp.historic_ohlcv(self.pair, self.t_frame).set_index('date')
        return h_data

    def __combine_data(self, c_data: DataFrame):
        c_data = c_data.copy().set_index('date')
        self.data = self.data.append(c_data[c_data.index.isin(self.data.index) == False], verify_integrity=True)

    def __prepare_data(self, df: DataFrame, frequency: str):
        df = df.copy()
        df['mean'] = (df['low'] + df['high']) / 2
        df['step_ahead'] = df['mean'].shift(self.steps)
        df = df.dropna()

        x = df.copy().drop('step_ahead', axis=1).asfreq(freq=frequency).interpolate()

        y = df.copy().drop('open', axis=1).drop('high', axis=1).drop('low', axis=1).drop('volume', axis=1).drop('mean',
                                                                                                                axis=1).drop(
            'close', axis=1).asfreq(freq=frequency).interpolate()
        return x, y

    def __get_train_data(self, x, y, current_prediction_index):
        if current_prediction_index - self.train_size < 0:
            raise Exception('Not enough data in history. Please first download enough data')

        train_x = x.iloc[current_prediction_index - self.train_size:current_prediction_index, :].copy()
        train_y = y.iloc[current_prediction_index - self.train_size:current_prediction_index, :].copy()

        fix_scaler = FixedScaler(train_x['mean'].min(), train_x['mean'].max(), train_x['volume'].min(),
                                 train_x['volume'].max())

        def scale(df):
            nonlocal fix_scaler
            df['volume'] = df['volume'].apply(lambda v: fix_scaler.scale(v))

        scale(train_x)
        return train_x, train_y, scale


class FixedScaler(object):

    def __init__(self, min_in, max_in, min_out, max_out):
        self.min_in = min_in
        self.max_in = max_in
        self.min_out = min_out
        self.max_out = max_out

    def scale(self, value):
        value_std = (value - self.min_out) / (self.max_out - self.min_out)
        return value_std * (self.max_in - self.min_in) + self.min_in