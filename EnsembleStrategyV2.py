from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import logging
from pandas import DataFrame
from freqtrade.resolvers import StrategyResolver
from itertools import combinations
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime

logger = logging.getLogger(__name__)


STRATEGIES = [
    "CombinedBinHAndCluc",
    "CombinedBinHAndClucV2",
    "CombinedBinHAndClucV5",
    "CombinedBinHAndClucV7",
    "CombinedBinHAndClucV8",
    "SMAOffset",
    "SMAOffsetV2",
    "SMAOffsetProtectOptV0",
    "SMAOffsetProtectOptV1",
    "NostalgiaForInfinityV1",
    "NostalgiaForInfinityV2",
    "NostalgiaForInfinityV3",
    "NostalgiaForInfinityV4",
    "NostalgiaForInfinityV5",
    "NostalgiaForInfinityV7",
]

STRAT_COMBINATIONS = reduce(
    lambda x, y: list(combinations(STRATEGIES, y)) + x, range(len(STRATEGIES)+1), []
)

MAX_COMBINATIONS = len(STRAT_COMBINATIONS) - 1


class EnsembleStrategyV2(IStrategy):
    loaded_strategies = {}

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    informative_timeframe = '1h'
    buy_mean_threshold = DecimalParameter(0.0, 1, default=0.032, load=True)
    sell_mean_threshold = DecimalParameter(0.0, 1, default=0.059, load=True)
    buy_strategies = IntParameter(0, MAX_COMBINATIONS, default=30080, load=True)
    sell_strategies = IntParameter(0, MAX_COMBINATIONS, default=21678, load=True)

    # Buy hyperspace params:
    buy_params = {
        "buy_mean_threshold": 0.032,
        "buy_strategies": 30080,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_mean_threshold": 0.059,
        "sell_strategies": 21678,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.22,
        "37": 0.073,
        "86": 0.016,
        "195": 0
    }

    # Stoploss:
    stoploss = -0.148

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.068
    trailing_stop_positive_offset = 0.081
    trailing_only_offset_is_reached = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info(f"Buy stratrategies: {STRAT_COMBINATIONS[self.buy_strategies.value]}")
        logger.info(f"Sell stratrategies: {STRAT_COMBINATIONS[self.sell_strategies.value]}")

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_strategy(self, strategy_name):
        strategy = self.loaded_strategies.get(strategy_name)
        if not strategy:
            config = self.config
            config["strategy"] = strategy_name
            strategy = StrategyResolver.load_strategy(config)

        strategy.dp = self.dp
        strategy.wallets = self.wallets
        self.loaded_strategies[strategy_name] = strategy
        return strategy

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # TODO: move all strats signals to here, add mean and difference mean for buy and sell
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        strategies = STRAT_COMBINATIONS[self.buy_strategies.value]
        for strategy_name in strategies:
            strategy = self.get_strategy(strategy_name)
            try:
                strategy_indicators = strategy.advise_indicators(dataframe, metadata)
                dataframe[f"strat_buy_signal_{strategy_name}"] = strategy.advise_buy(
                    strategy_indicators, metadata
                )["buy"]
            except Exception:
                pass

        dataframe['buy'] = (
            dataframe.filter(like='strat_buy_signal_').fillna(0).mean(axis=1) > self.buy_mean_threshold.value
        ).astype(int)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0
        return dataframe

    def custom_sell(
        self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if (last_candle is not None):
            strategies = STRAT_COMBINATIONS[self.sell_strategies.value]
            metadata = {"pair": pair}
            for strategy_name in strategies:
                strategy = self.get_strategy(strategy_name)
                try:
                    strategy_indicators = strategy.advise_indicators(dataframe, metadata)
                    dataframe[f"strat_sell_signal_{strategy_name}"] = strategy.advise_sell(
                        strategy_indicators, metadata
                    )["sell"]
                except Exception:
                    pass

            dataframe['sell'] = (
                dataframe.filter(like='strat_sell_signal_').fillna(0).mean(axis=1) > self.sell_mean_threshold.value
            ).astype(int)
            last_candle = dataframe.iloc[-1].squeeze()
            return last_candle.sell

        return None
