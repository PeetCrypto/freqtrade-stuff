from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import logging
from pandas import DataFrame
from freqtrade.resolvers import StrategyResolver
from itertools import combinations
from functools import reduce

logger = logging.getLogger(__name__)

"""
Hyperoptimizedd using OnlyProfitHyperOptLoss
======================================================= SELL REASON STATS ========================================================
|        Sell Reason |   Sells |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|        sell_signal |     507 |    333     0   174  65.7 |           0.13 |          67.95 |           253.527 |          16.99 |
| trailing_stop_loss |     103 |    103     0     0   100 |           4.88 |         502.33 |          2764.21  |         125.58 |
|                roi |      37 |     35     2     0   100 |           7.93 |         293.51 |          1247.71  |          73.38 |
|          stop_loss |      12 |      0     0    12     0 |         -20.46 |        -245.51 |         -1139.88  |         -61.38 |
====================================================== LEFT OPEN TRADES REPORT ======================================================
|   Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|--------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|  TOTAL |      0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2021-05-01 00:00:00 |
| Backtesting to        | 2021-05-31 15:30:00 |
| Max open trades       | 4                   |
|                       |                     |
| Total trades          | 659                 |
| Starting balance      | 1000.000 USDT       |
| Final balance         | 4125.571 USDT       |
| Absolute profit       | 3125.571 USDT       |
| Total profit %        | 312.56%             |
| Trades per day        | 21.97               |
| Avg. stake amount     | 533.922 USDT        |
| Total trade volume    | 351854.681 USDT     |
|                       |                     |
| Best Pair             | MATIC/USDT 180.8%   |
| Worst Pair            | STORJ/USDT -20.27%  |
| Best trade            | DOT/USDT 24.18%     |
| Worst trade           | ZEC/USDT -20.46%    |
| Best day              | 793.247 USDT        |
| Worst day             | -198.590 USDT       |
| Days win/draw/lose    | 26 / 0 / 5          |
| Avg. Duration Winners | 0:41:00             |
| Avg. Duration Loser   | 1:52:00             |
| Zero Duration Trades  | 3.64% (24)          |
| Rejected Buy signals  | 45400               |
|                       |                     |
| Min balance           | 1011.385 USDT       |
| Max balance           | 4125.571 USDT       |
| Drawdown              | 186.4%              |
| Drawdown              | 824.540 USDT        |
| Drawdown high         | 1100.474 USDT       |
| Drawdown low          | 275.934 USDT        |
| Drawdown Start        | 2021-05-19 01:20:00 |
| Drawdown End          | 2021-05-19 12:50:00 |
| Market change         | -28.01%             |
===============================================
"""

# DO NOT MODIFY THE STRATEGY LIST
# You'll need to run hyperopt to find the best strategy combination for buy/sell.
# Also, make sure you have all strategies listed here in user_data/strategies
STRATEGIES = [
    "CombinedBinHAndCluc",
    "CombinedBinHAndClucV2",
    "CombinedBinHAndClucV5",
    "CombinedBinHAndClucV6H",
    "CombinedBinHAndClucV7",
    "CombinedBinHAndClucV8",
    "CombinedBinHAndClucV8Hyper",
    "SMAOffset",
    "SMAOffsetV2",
    "NostalgiaForInfinityV1",
    "NostalgiaForInfinityV2",
]

STRAT_COMBINATIONS = reduce(
    lambda x, y: list(combinations(STRATEGIES, y)) + x, range(len(STRATEGIES)+1), []
)


class EnsembleStrategyV1(IStrategy):
    loaded_strategies = {}
    buy_mean_threshold = DecimalParameter(0.0, 1, default=0.5, load=True)
    sell_mean_threshold = DecimalParameter(0.0, 1, default=0.5, load=True)
    buy_strategies = IntParameter(0, len(STRAT_COMBINATIONS), default=0, load=True)
    sell_strategies = IntParameter(0, len(STRAT_COMBINATIONS), default=0, load=True)

    # Buy hyperspace params:
    buy_params = {
        "buy_mean_threshold": 0.124,
        "buy_strategies": 1440,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_mean_threshold": 0.791,
        "sell_strategies": 1654,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.242,
        "28": 0.046,
        "68": 0.035,
        "137": 0
    }

    # Stoploss:
    stoploss = -0.203

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.041
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    informative_timeframe = '1h'

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info(f"Buy stratrategies: {STRAT_COMBINATIONS[self.buy_strategies.value]}")
        logger.info(f"Sell stratrategies: {STRAT_COMBINATIONS[self.sell_strategies.value]}")

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_strategy(self, strategy_name):
        cached_strategy = self.loaded_strategies.get(strategy_name)
        if cached_strategy:
            cached_strategy.dp = self.dp
            return cached_strategy

        config = self.config
        config["strategy"] = strategy_name
        strategy = StrategyResolver.load_strategy(config)
        strategy.dp = self.dp
        self.loaded_strategies[strategy_name] = strategy
        return strategy

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        strategies = STRAT_COMBINATIONS[self.buy_strategies.value]
        for strategy_name in strategies:
            strategy = self.get_strategy(strategy_name)
            strategy_indicators = strategy.advise_indicators(dataframe, metadata)
            dataframe[f"strat_buy_signal_{strategy_name}"] = strategy.advise_buy(
                strategy_indicators, metadata
            )["buy"]

        dataframe['buy'] = (
            dataframe.filter(like='strat_buy_signal_').mean(axis=1) > self.buy_mean_threshold.value
        ).astype(int)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        strategies = STRAT_COMBINATIONS[self.sell_strategies.value]
        for strategy_name in strategies:
            strategy = self.get_strategy(strategy_name)
            strategy_indicators = strategy.advise_indicators(dataframe, metadata)
            dataframe[f"strat_sell_signal_{strategy_name}"] = strategy.advise_sell(
                strategy_indicators, metadata
            )["sell"]

        dataframe['sell'] = (
            dataframe.filter(like='strat_sell_signal_').mean(axis=1) > self.sell_mean_threshold.value
        ).astype(int)
        return dataframe
