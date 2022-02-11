from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, stoploss_from_open
import logging
from pandas import DataFrame
from freqtrade.resolvers import StrategyResolver
from itertools import combinations
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime

logger = logging.getLogger(__name__)

STRATEGIES = [
    "SMAOffsetV2",
    "MADisplaceV3"
]

STRAT_COMBINATIONS = reduce(
    lambda x, y: list(combinations(STRATEGIES, y)) + x, range(len(STRATEGIES) + 1), []
)

MAX_COMBINATIONS = len(STRAT_COMBINATIONS) - 2


class Chained(IStrategy):
    loaded_strategies = {}
    informative_timeframe = "1h"
    buy_action_diff_threshold = DecimalParameter(0, 1, default=0, decimals=2, optimize=True, load=True)
    buy_strategies = IntParameter(0, MAX_COMBINATIONS, default=0, optimize=True, load=True)

    # trailing stoploss hyperopt parameters
    # hard stoploss profit
    sell_HSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, optimize=True, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    sell_PF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, optimize=True, load=True)
    sell_SL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, optimize=True, load=True)

    # profit threshold 2, SL_2 is used
    sell_PF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, optimize=True, load=True)
    sell_SL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, optimize=True, load=True)

    stoploss = -0.99  # effectively disabled.
    sell_profit_offset = 0.001  # it doesn't meant anything, just to guarantee there is a minimal profit.
    use_sell_signal = False
    ignore_roi_if_buy_signal = False
    sell_profit_only = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    minimal_roi = {
        "0": 100.0
    }

    buy_params = {}
    sell_params = {}

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 100,
            "trade_limit": 4,
            "stop_duration_candles": 10,
            "only_per_pair": True
        },
    ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info(f"Buy stratrategies: {STRAT_COMBINATIONS[self.buy_strategies.value]}")

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
            dataframe.filter(like='strat_buy_signal_').mean(axis=1) > self.buy_action_diff_threshold.value
        ).astype(int)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0
        return dataframe

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:
        """
        new custom stoploss, both hard and trailing functions. Trailing stoploss first rises at a slower
        rate than the current rate until a profit threshold is reached, after which it rises at a constant
        percentage as per a normal trailing stoploss. This allows more margin for pull-backs during a rise.
        """

        # hard stoploss profit
        HSL = self.sell_HSL.value
        PF_1 = self.sell_PF_1.value
        SL_1 = self.sell_SL_1.value
        PF_2 = self.sell_PF_2.value
        SL_2 = self.sell_SL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.
        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL

        if (current_profit > PF_1):
            stoploss = stoploss_from_open(sl_profit, current_profit)
        else:
            stoploss = stoploss_from_open(HSL, current_profit)

        return stoploss or stoploss_from_open(HSL, current_profit) or 1
