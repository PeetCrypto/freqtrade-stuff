# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open
from datetime import datetime


# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
	df = dataframe.copy()
	ema1 = ta.EMA(df, timeperiod=ema_length)
	ema2 = ta.EMA(df, timeperiod=ema2_length)
	emadif = (ema1 - ema2) / df['close'] * 100
	return emadif


"""
======================================================= SELL REASON STATS ========================================================
|        Sell Reason |   Sells |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|        sell_signal |     392 |    159     0   233  40.6 |          -0.45 |        -178.25 |          -892.121 |         -59.42 |
| trailing_stop_loss |     187 |    187     0     0   100 |           3.53 |         659.86 |          3302.61  |         219.95 |
====================================================== LEFT OPEN TRADES REPORT ======================================================
|   Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|--------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|  TOTAL |      0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
=============== SUMMARY METRICS ================
| Metric                 | Value               |
|------------------------+---------------------|
| Backtesting from       | 2021-10-12 00:00:00 |
| Backtesting to         | 2021-11-12 00:00:00 |
| Max open trades        | 3                   |
|                        |                     |
| Total/Daily Avg Trades | 579 / 18.68         |
| Starting balance       | 1000.000 USDT       |
| Final balance          | 3410.494 USDT       |
| Absolute profit        | 2410.494 USDT       |
| Total profit %         | 241.05%             |
| Trades per day         | 18.68               |
| Avg. daily profit %    | 7.78%               |
| Avg. stake amount      | 500.000 USDT        |
| Total trade volume     | 289500.000 USDT     |
|                        |                     |
| Best Pair              | QRDO/USDT 35.47%    |
| Worst Pair             | DAG/USDT -12.07%    |
| Best trade             | ARX/USDT 8.93%      |
| Worst trade            | ATOM/USDT -11.35%   |
| Best day               | 235.119 USDT        |
| Worst day              | -36.114 USDT        |
| Days win/draw/lose     | 26 / 0 / 5          |
| Avg. Duration Winners  | 1:41:00             |
| Avg. Duration Loser    | 3:10:00             |
| Rejected Buy signals   | 217074              |
|                        |                     |
| Min balance            | 955.214 USDT        |
| Max balance            | 3410.494 USDT       |
| Drawdown               | 22.67%              |
| Drawdown               | 113.471 USDT        |
| Drawdown high          | 2385.361 USDT       |
| Drawdown low           | 2271.890 USDT       |
| Drawdown Start         | 2021-11-10 17:55:00 |
| Drawdown End           | 2021-11-10 23:05:00 |
| Market change          | 0%                  |
================================================


Epoch details:

*    5/90:    579 trades. 346/0/233 Wins/Draws/Losses. Avg profit   0.83%. Median profit   0.54%. Total profit  2410.49362831 USDT ( 241.05%). Avg duration 2:17:00 min. Objective: -112.66357


    # Buy hyperspace params:
    buy_params = {
        "ADX_thresold": 40,
        "BB_length": 20,
        "BB_multifactor": 2,
        "KC_length": 25,
        "KC_multifactor": 1.5,
        "RSI_overbought": 45,
        "use_true_range": True,
    }

    # Sell hyperspace params:
    sell_params = {
        "pHSL": -0.08,  # value loaded from strategy
        "pPF_1": 0.016,  # value loaded from strategy
        "pPF_2": 0.08,  # value loaded from strategy
        "pSL_1": 0.011,  # value loaded from strategy
        "pSL_2": 0.04,  # value loaded from strategy
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.3
    }

    # Stoploss:
    stoploss = -0.99  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.03  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy
    """
"""
======================================================= SELL REASON STATS ========================================================
|        Sell Reason |   Sells |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|        sell_signal |     457 |    196     0   261  42.9 |          -0.57 |        -258.56 |         -1294.08  |         -86.19 |
| trailing_stop_loss |     261 |    261     0     0   100 |           3.57 |         931.7  |          4663.14  |         310.57 |
|         force_sell |       2 |      0     0     2     0 |          -0.64 |          -1.28 |            -6.389 |          -0.43 |
======================================================= LEFT OPEN TRADES REPORT =======================================================
|     Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|----------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| MHC/USDT |      1 |          -0.36 |          -0.36 |            -1.801 |          -0.18 |        1:30:00 |     0     0     1     0 |
| XLM/USDT |      1 |          -0.92 |          -0.92 |            -4.588 |          -0.46 |        1:05:00 |     0     0     1     0 |
|    TOTAL |      2 |          -0.64 |          -1.28 |            -6.389 |          -0.64 |        1:18:00 |     0     0     2     0 |
=============== SUMMARY METRICS ================
| Metric                 | Value               |
|------------------------+---------------------|
| Backtesting from       | 2021-10-12 00:00:00 |
| Backtesting to         | 2021-11-12 00:00:00 |
| Max open trades        | 3                   |
|                        |                     |
| Total/Daily Avg Trades | 720 / 23.23         |
| Starting balance       | 1000.000 USDT       |
| Final balance          | 4362.668 USDT       |
| Absolute profit        | 3362.668 USDT       |
| Total profit %         | 336.27%             |
| Trades per day         | 23.23               |
| Avg. daily profit %    | 10.85%              |
| Avg. stake amount      | 500.000 USDT        |
| Total trade volume     | 360000.000 USDT     |
|                        |                     |
| Best Pair              | XNL/USDT 45.69%     |
| Worst Pair             | DAG/USDT -8.3%      |
| Best trade             | XNL/USDT 19.72%     |
| Worst trade            | DAPPT/USDT -9.72%   |
| Best day               | 229.122 USDT        |
| Worst day              | -8.923 USDT         |
| Days win/draw/lose     | 29 / 0 / 3          |
| Avg. Duration Winners  | 1:43:00             |
| Avg. Duration Loser    | 3:23:00             |
| Rejected Buy signals   | 377484              |
|                        |                     |
| Min balance            | 968.162 USDT        |
| Max balance            | 4376.206 USDT       |
| Drawdown               | 18.82%              |
| Drawdown               | 94.205 USDT         |
| Drawdown high          | 3312.883 USDT       |
| Drawdown low           | 3218.678 USDT       |
| Drawdown Start         | 2021-11-10 15:55:00 |
| Drawdown End           | 2021-11-10 23:40:00 |
| Market change          | 0%                  |
================================================


Epoch details:

   462/684:    720 trades. 457/0/263 Wins/Draws/Losses. Avg profit   0.93%. Median profit   0.69%. Total profit  3362.66761261 USDT ( 336.27%). Avg duration 2:19:00 min. Objective: -147.06218


    # Buy hyperspace params:
    buy_params = {
        "ADX_thresold": 33,
        "BB_length": 22,
        "BB_multifactor": 1.5,
        "KC_length": 28,
        "KC_multifactor": 1,
        "RSI_overbought": 45,
        "use_true_range": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "pHSL": -0.08,  # value loaded from strategy
        "pPF_1": 0.016,  # value loaded from strategy
        "pPF_2": 0.08,  # value loaded from strategy
        "pSL_1": 0.011,  # value loaded from strategy
        "pSL_2": 0.04,  # value loaded from strategy
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.3
    }

    # Stoploss:
    stoploss = -0.99  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.03  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy
    """

class SqueezeMomentum(IStrategy):
	INTERFACE_VERSION = 2

	# Buy hyperspace params:
	buy_params = {
		'BB_length': 20,
		'BB_multifactor': 2.0,
		'KC_length': 11,
		'KC_multifactor': 1.5,
		'use_true_range': True,
		'RSI_overbought': 60,
		'ADX_thresold': 33,
	}

	# Sell hyperspace params:
	sell_params = {
	}

	# ROI table:  # value loaded from strategy
	minimal_roi = {
		"0": 0.3
	}

	# Stoploss:
	stoploss = -0.99  # value loaded from strategy

	# Trailing stop:
	trailing_stop = True  # value loaded from strategy
	trailing_stop_positive = 0.005  # value loaded from strategy
	trailing_stop_positive_offset = 0.03  # value loaded from strategy
	trailing_only_offset_is_reached = True  # value loaded from strategy

	use_custom_stoploss = False

	# Sell signal
	use_sell_signal = True
	sell_profit_only = False
	sell_profit_offset = 0.01
	ignore_roi_if_buy_signal = False
	process_only_new_candles = True
	startup_candle_count = 30

	# Parameters
	BB_length = IntParameter(10, 30, default=buy_params['BB_length'], space='buy', optimize=True)
	BB_multifactor = CategoricalParameter([0.5, 1, 1.5, 2, 2.5, 3, 3.5], default=buy_params['BB_multifactor'], space='buy', optimize=True)
	KC_length = IntParameter(10, 30, default=buy_params['KC_length'], space='buy', optimize=True)
	KC_multifactor = CategoricalParameter([0.5, 1, 1.5, 2, 2.5, 3, 3.5], default=buy_params['KC_multifactor'], space='buy', optimize=True)
	use_true_range = BooleanParameter(default=buy_params['use_true_range'], space='buy', optimize=True)

	# Guards
	RSI_overbought = CategoricalParameter([45, 50, 55, 60, 65], default=buy_params['RSI_overbought'], space='buy', optimize=True)
	ADX_thresold = CategoricalParameter([15, 20, 25, 30, 33, 35, 40, 45, 50], default=buy_params['ADX_thresold'], space='buy', optimize=True)

	## Trailing params

	# hard stoploss profit
	pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
	# profit threshold 1, trigger point, SL_1 is used
	pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
	pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

	# profit threshold 2, SL_2 is used
	pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
	pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

	# Optimal timeframe for the strategy
	timeframe = '5m'

	## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
	def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
						current_rate: float, current_profit: float, **kwargs) -> float:

		# hard stoploss profit
		HSL = self.pHSL.value
		PF_1 = self.pPF_1.value
		SL_1 = self.pSL_1.value
		PF_2 = self.pPF_2.value
		SL_2 = self.pSL_2.value

		# For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
		# between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
		# rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

		if (current_profit > PF_2):
			sl_profit = SL_2 + (current_profit - PF_2)
		elif (current_profit > PF_1):
			sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
		else:
			sl_profit = HSL

		# Only for hyperopt invalid return
		if (sl_profit >= current_profit):
			return -0.99

		return stoploss_from_open(sl_profit, current_profit)

	def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		"""
        //
        // @author LazyBear
        // List of all my indicators: https://www.tradingview.com/v/4IneGo8h/
        //

        // Calculate BB
        source = close
        basis = sma(source, length)
        dev = multKC * stdev(source, length)
        upperBB = basis + dev
        lowerBB = basis - dev

        // Calculate KC
        ma = sma(source, lengthKC)
        range = useTrueRange ? tr : (high - low)
        rangema = sma(range, lengthKC)
        upperKC = ma + rangema * multKC
        lowerKC = ma - rangema * multKC

        sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
        noSqz  = (sqzOn == false) and (sqzOff == false)

        val = linreg(source  -  avg(avg(highest(high, lengthKC), lowest(low, lengthKC)),sma(close,lengthKC)),
                    lengthKC,0)

        bcolor = iff( val > 0,
                    iff( val > nz(val[1]), lime, green),
                    iff( val < nz(val[1]), red, maroon))
        scolor = noSqz ? blue : sqzOn ? black : gray
        plot(val, color=bcolor, style=histogram, linewidth=4)
        plot(0, color=scolor, style=cross, linewidth=2)
        """

		if self.use_true_range.value:
			dataframe[f'range'] = ta.TRANGE(dataframe)
		else:
			dataframe[f'range'] = dataframe['high'] - dataframe['low']

		for val in self.BB_length.range:
			# BB
			dataframe[f'ma_{val}'] = ta.SMA(dataframe, val)
			dataframe[f'stdev_{val}'] = ta.STDDEV(dataframe, val)
			# KC
			dataframe[f'rangema_{val}'] = ta.SMA(dataframe[f'range'], val)

			# Linreg
			dataframe[f'hh_close_{val}'] = ta.MAX(dataframe['high'], val)
			dataframe[f'll_close_{val}'] = ta.MIN(dataframe['low'], val)
			dataframe[f'avg_hh_ll_{val}'] = (dataframe[f'hh_close_{val}'] + dataframe[f'll_close_{val}']) / 2
			dataframe[f'avg_close_{val}'] = ta.SMA(dataframe['close'], val)
			dataframe[f'avg_{val}'] = (dataframe[f'avg_hh_ll_{val}'] + dataframe[f'avg_close_{val}']) / 2
			dataframe[f'val_{val}'] = ta.LINEARREG(dataframe['close'] - dataframe[f'avg_{val}'], val, 0)

			# min val
			dataframe[f'val_min_{val}'] = ta.MIN(dataframe[f'val_{val}'], 50)
			# max val
			dataframe[f'val_max_{val}'] = ta.MAX(dataframe[f'val_{val}'], 50)
			# stdev val
			dataframe[f'val_stdev_{val}'] = ta.STDDEV(dataframe[f'val_{val}'], 50)
			# average val
			dataframe[f'val_avg_{val}'] = ta.SMA(dataframe[f'val_{val}'], 50)

		for val in self.KC_length.range:
			# BB
			dataframe[f'ma_{val}'] = ta.SMA(dataframe, val)
			dataframe[f'stdev_{val}'] = ta.STDDEV(dataframe, val)
			# KC
			dataframe[f'rangema_{val}'] = ta.SMA(dataframe[f'range'], val)

		# RSI
		dataframe['rsi'] = ta.RSI(dataframe)

		# EMA
		dataframe['ema_50'] = ta.EMA(dataframe, 50)
		dataframe['ema_200'] = ta.EMA(dataframe, 200)

		# ADX
		dataframe['adx'] = ta.ADX(dataframe, 14)

		return dataframe

	def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		bb_length = self.BB_length.value
		mult = self.BB_multifactor.value
		kc = self.KC_multifactor.value
		kc_length = self.KC_length.value

		is_sqzOn = (
				(dataframe[f'ma_{bb_length}'] - dataframe[f'stdev_{bb_length}'] * mult > dataframe[f'ma_{kc_length}'] - dataframe[f'rangema_{kc_length}'] * kc) &
				(dataframe[f'ma_{bb_length}'] + dataframe[f'stdev_{bb_length}'] * mult < dataframe[f'ma_{kc_length}'] + dataframe[f'rangema_{kc_length}'] * kc)
		)

		is_sqzOff = (
				(dataframe[f'ma_{bb_length}'] - dataframe[f'stdev_{bb_length}'] * mult < dataframe[f'ma_{kc_length}'] - dataframe[f'rangema_{kc_length}'] * kc) &
				(dataframe[f'ma_{bb_length}'] + dataframe[f'stdev_{bb_length}'] * mult > dataframe[f'ma_{kc_length}'] + dataframe[f'rangema_{kc_length}'] * kc)
		)

		# is_noSqz = (
		#        ((not is_sqzOn) & (not is_sqzOff))
		# )

		dataframe.loc[
			(
				# (dataframe[f'sqzOn_{self.KC_length.value}_{self.KC_multifactor.value}'] == 1) &
				# (dataframe[f'val_{self.BB_length.value}'] > 0) &
				# (dataframe[f'val_{self.BB_length.value}'].shift(1) < 0) &
				# (dataframe[f'sqzOn_{self.KC_length.value}_{self.KC_multifactor.value}'].shift(1) == 1) &
                    (is_sqzOff) &
                    (dataframe[f'val_{self.BB_length.value}'].shift(2) > dataframe[f'val_{self.BB_length.value}'].shift(1)) &
                    (dataframe[f'val_{self.BB_length.value}'].shift(1) < dataframe[f'val_{self.BB_length.value}']) &
                    # (dataframe[f'val_{self.BB_length.value}'].shift(1) == dataframe[f'val_min_{self.BB_length.value}']) &
                    # (dataframe[f'val_{self.BB_length.value}'].shift(1) < dataframe[f'val_avg_{self.BB_length.value}'] -  dataframe[f'val_stdev_{self.BB_length.value}']) &
                    (dataframe[f'val_{self.BB_length.value}'] < 0) &
                    (dataframe['adx'] > self.ADX_thresold.value) &
                    # (dataframe[f'sqzOn_{self.KC_length.value}_{self.KC_multifactor.value}'].rolling(10).sum() < 3) &
                    (dataframe['rsi'] < self.RSI_overbought.value) &
                    # (dataframe['close'] > dataframe['ema_50']) &
                    # (dataframe['ema_50'] > dataframe['ema_200']) &
                    (dataframe['volume'] > 0)
			),
			'buy'] = 1
		return dataframe

	def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		dataframe.loc[
			(
					(dataframe[f'val_{self.BB_length.value}'].shift(2) < dataframe[f'val_{self.BB_length.value}'].shift(1)) &
					(dataframe[f'val_{self.BB_length.value}'].shift(1) > dataframe[f'val_{self.BB_length.value}']) &
					(dataframe[f'val_{self.BB_length.value}'].shift(1) == dataframe[f'val_max_{self.BB_length.value}']) &
					(dataframe[f'val_{self.BB_length.value}'] > 0) &

(dataframe['volume'] > 0)
			),
			'sell'] = 1
		dataframe.to_csv('user_data/csvs/%s_%s.csv' % (self.__class__.__name__, metadata["pair"].replace("/", "_")))

		return dataframe


