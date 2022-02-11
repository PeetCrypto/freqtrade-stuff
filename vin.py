from freqtrade.strategy import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
import logging
import numpy as np
from pandas import DataFrame, Series, to_numeric
from functools import reduce
from datetime import datetime, timedelta
import locale
locale.setlocale(category=locale.LC_ALL, locale='')
log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    def version(self) -> str:
        return 'v1.1.1'

    write_to_csv = False
    df_csv = './user_data/df.csv'
    min_candle_vol: int = 0
    custom_buy_info = {}
    max_concurrent_buy_signals_check = True

    minimal_roi = {"0": 100}
    stoploss = -1
    stoploss_on_exchange = False
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    startup_candle_count: int = 90

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 36
            }
        ]

    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators_sell(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['green'] = (df['close'] - df['open']).ge(0)
        df['bodysize'] = (df['close'] / df['open']).where(df['green'], df['open'] / df['close'])
        hi_adj = df['close'].where(df['green'], df['open']) + (df['high'] - df['close']).where(df['green'], (df['high'] - df['open'])) / df['bodysize'].pow(0.25)
        lo_adj = df['open'].where(df['green'], df['close']) - (df['open'] - df['low']).where(df['green'], (df['close'] - df['low'])) / df['bodysize'].pow(0.25)
        df['hlc3_adj'] = (hi_adj + lo_adj + df['close']) / 3
        df['lc2_adj'] = (lo_adj + df['close']) / 2
        df['hc2_adj'] = (hi_adj + df['close']) / 2
        df[f"candle_count_{self.startup_candle_count}"] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()
        df_closechange = df['close'] - df['close'].shift(1)
        s = (1, 2, 3)
        for i in s:
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=i).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=i).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df['streak_s_min'] = df[[f"streak_{i}" for i in s]].min(axis=1)
        df['streak_s_min_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() - df['streak_s_min'].abs().to_numpy()]
        df['streak_s_max'] = df[[f"streak_{i}" for i in s]].max(axis=1)
        df.drop(columns=[f"streak_{i}" for i in s], inplace=True)
        df_closechange = df['close'] - df['close'].shift(1)
        i = 12
        df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=i).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=i).sum().lt(0), -1, 0))
        df[f"streak_h"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df.drop(columns=['updown'], inplace=True)
        df = self.populate_indicators_buy(df, metadata)
        df = self.populate_indicators_sell(df, metadata)
        if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_csv:
            ef = df[['date', 'open', 'high', 'low', 'close', 'volume', 'bodysize', 'hlc3_adj', 'lc2_adj', 'hc2_adj', 'streak_s_min', 'streak_s_max']]
            ef['pair'] = metadata['pair']
            with open(self.df_csv, 'a') as f:
                ef.to_csv(f, sep=';', header=f.tell()==0, index=False)
        return df

    def fill_custom_buy_info(self, df:DataFrame, metadata: dict):
        df_buy: DataFrame = df.loc[df['buy'], ['date', 'buy_tag']]
        for index, row in df_buy.iterrows():
            buy_date = row['date']
            if buy_date not in self.custom_buy_info:
                self.custom_buy_info[buy_date] = {}
                self.custom_buy_info[buy_date]['buy_signals'] = 1
            else:
                self.custom_buy_info[buy_date]['buy_signals'] += 1
            self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
        return None

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy'] = False
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'sell'] = False
        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
            return None

    def bot_loop_start(self, **kwargs) -> None:
        self.min_candle_vol = self.config['stake_amount']
        if self.write_to_csv:
            with open(self.df_csv, 'w') as f:
                pass
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        buy_candle_date = df['date'].iloc[-1]
        d = buy_candle_date.strftime('%Y-%m-%d %H:%M')
        try:
            buy_info = self.custom_buy_info[buy_candle_date]
            buy_tag = buy_info[pair]
            buy_signal_count = buy_info['buy_signals']
            if self.max_concurrent_buy_signals_check:
                pairs = len(self.dp.current_whitelist())
                max_concurrent_buy_signals = max(int(pairs * 0.08), 2)
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"{d} confirm_trade_entry: Cancel buy for pair {pair} with buy tag {buy_tag}. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False
            log.info(f"{d} confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tag} and {buy_signal_count} concurrent buy signals.")
        except:
            log.warning(f"{d} confirm_trade_entry: No buy info for pair {pair}.")
            return False
        return True

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        buy_candle_date = df['date'].iloc[-1]
        d = buy_candle_date.strftime('%Y-%m-%d %H:%M')
        try:
            buy_info = self.custom_buy_info[buy_candle_date]
            buy_signal_count = buy_info['buy_signals']
            if self.max_concurrent_buy_signals_check:
                pairs = len(self.dp.current_whitelist())
                max_concurrent_buy_signals = max(int(pairs * 0.04), 1)
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"{d} confirm_trade_exit: Cancel sell for pair {pair}. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False
        except:
            return True
        return True

def vws(df: DataFrame, length: int) -> Series:
    mf: Series = df['hlc3_adj'] * df['volume']
    mfp = mf.where(df['hlc3_adj'].pct_change().gt(0), 0).rolling(window=length, min_periods=1).sum()
    mfn = mf.where(df['hlc3_adj'].pct_change().lt(0), 0).rolling(window=length, min_periods=1).sum()
    return 100 * (mfp / (mfp + mfn))

class ViNBuyPct(ViN):
    buy_lookback_range = range(29, 74)
    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        for i in self.buy_lookback_range:
            df[f"pctchange_{i}"] = df['close'].pct_change(periods=i)
            pctchange_mean = df[f"pctchange_{i}"].rolling(window=i, min_periods=i).mean()
            pctchange_std = df[f"pctchange_{i}"].rolling(window=i, min_periods=i).std()
            df[f"bb_pctchange_{i}_up"] = pctchange_mean + 2 * pctchange_std
            df[f"bb_pctchange_{i}_lo"] = pctchange_mean - 2 * pctchange_std
            df = df.copy()
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag'] = ''
        for i in self.buy_lookback_range:
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 18),
                df['streak_s_min'].le(-1),
                df['streak_s_max'].between(-5, 0),
                df['streak_h'].ge(-19),
                df['streak_s_min'].ge(df['streak_h']),
                df['streak_s_min_change'].le(0.97),
                (df[f"pctchange_{i}"] / df[f"bb_pctchange_{i}_lo"]).between(1.01, 1.39),
                (df[f"bb_pctchange_{i}_up"] - df[f"bb_pctchange_{i}_lo"]).ge(0.02),
                (df['lc2_adj'] / df['close']).between(0.975, 0.995)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
        tag_begin = df['buy_tag'].str[:3]
        tag_end = df['buy_tag'].str[-3:-1]
        tag_begin_end = tag_begin + tag_end
        df.loc[:, 'buy'] = df['buy_tag'].ne('') & tag_begin_end.ne('29 73') # & tag_begin_end.ne('30 30') & tag_begin_end.ne('31 31') & tag_begin_end.ne('32 32') & ~to_numeric(tag_end).between(62, 69)
        df.loc[df['buy'], 'buy_tag'] = 'pct ' + tag_begin_end
        self.fill_custom_buy_info(df, metadata)
        # print(df.loc[df['buy'], ['date', 'volume', 'streak_s_min', 'streak_s_max', 'streak_h', 'streak_s_min_change', 'close', 'lc2_adj']])
        return df

class ViNBuyLc2(ViN):
    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag'] = ''
        buy_lookback_range = range(49, 91)
        for j in buy_lookback_range:
            lc2_low: DataFrame = df['lc2_adj'].rolling(window=j, min_periods=1).min()
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 8),
                (df['lc2_adj'].shift(1) / lc2_low.shift(2)).lt(0.99),
                (df['lc2_adj'] / lc2_low.shift(2)).gt(1)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{j} "
        tag_begin = df['buy_tag'].str[:3]
        tag_end = df['buy_tag'].str[-3:-1]
        tag_begin_end = tag_begin + tag_end
        df.loc[:, 'buy'] = df['buy_tag'].ne('') & tag_begin_end.ne('49 90')
        df.loc[df['buy'], 'buy_tag'] = 'lc2 ' + tag_begin_end
        self.fill_custom_buy_info(df, metadata)
        return df

class ViNBuyPctLc2(ViN):
    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag_pct'] = ''
        buy_lookback_range = range(29, 74)
        for i in buy_lookback_range:
            pctchange = df['close'].pct_change(periods=i)
            pctchange_mean = pctchange.rolling(window=i, min_periods=i).mean()
            pctchange_std = pctchange.rolling(window=i, min_periods=i).std()
            bb_pctchange_up = pctchange_mean + 2 * pctchange_std
            bb_pctchange_lo = pctchange_mean - 2 * pctchange_std
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 18),
                df['streak_s_min'].le(-1),
                df['streak_s_max'].between(-5, 0),
                df['streak_h'].ge(-19),
                df['streak_s_min'].ge(df['streak_h']),
                df['streak_s_min_change'].le(0.97),
                (pctchange / bb_pctchange_lo).between(1.01, 1.39),
                (bb_pctchange_up - bb_pctchange_lo).ge(0.02),
                (df['lc2_adj'] / df['close']).between(0.975, 0.995)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag_pct'] += f"{i} "
        tag_begin_pct = df['buy_tag_pct'].str[:3]
        tag_end_pct = df['buy_tag_pct'].str[-3:-1]
        tag_begin_end_pct = tag_begin_pct + tag_end_pct
        df.loc[:, 'buy_tag_lc2'] = ''
        buy_lookback_range = range(49, 91)
        for j in buy_lookback_range:
            lc2_low: DataFrame = df['lc2_adj'].rolling(window=j, min_periods=1).min()
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 8),
                (df['lc2_adj'].shift(1) / lc2_low.shift(2)).lt(0.99),
                (df['lc2_adj'] / lc2_low.shift(2)).gt(1)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag_lc2'] += f"{j} "
        tag_begin_lc2 = df['buy_tag_lc2'].str[:3]
        tag_end_lc2 = df['buy_tag_lc2'].str[-3:-1]
        tag_begin_end_lc2 = tag_begin_lc2 + tag_end_lc2
        buy_pct = df['buy_tag_pct'].ne('') & tag_begin_end_pct.ne('29 73') # & tag_begin_end_pct.ne('30 30') & tag_begin_end_pct.ne('31 31') & tag_begin_end_pct.ne('32 32') & ~to_numeric(tag_end_pct).between(62, 69)
        buy_lc2 = df['buy_tag_lc2'].ne('') & tag_begin_end_lc2.ne('49 90')
        df.loc[:, 'buy'] = buy_pct | buy_lc2
        df.loc[df['buy'], 'buy_tag'] = ('pct ' + tag_begin_end_pct + ' ').where(buy_pct, '')
        df.loc[df['buy'], 'buy_tag'] += ('lc2 ' + tag_begin_end_lc2).where(buy_lc2, '')
        df.loc[df['buy'], 'buy_tag'] = df['buy_tag'].str.strip()
        self.fill_custom_buy_info(df, metadata)
        return df

class ViNBuyVws(ViN):
    buy_lookback_range = range(10, 91)
    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        ef = df[['close', 'hlc3_adj', 'volume']].reset_index()
        for i in self.buy_lookback_range:
            df[f"vws_{i}"] = vws(df, length=i)
            df[f"hlc3_corr_{i}"] = ef['index'].rolling(window=i, min_periods=i).corr(ef['hlc3_adj'], method='spearman')
            df = df.copy()
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag'] = ''
        # df.loc[:, 'buy_tag_sum'] = 0
        for i in self.buy_lookback_range:
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 18),
                df[f"vws_{i}"].between(2, 18), #2, pow(i, 0.67)),
                df[f"hlc3_corr_{i}"].between(-0.999, -0.999 + pow(i, 0.5) / 1000),
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
            # df.loc[buy, 'buy_tag_sum'] += i
        # tag_begin = df['buy_tag'].str[:3]
        # tag_end= df['buy_tag'].str[-3:-1]
        # tag_begin_end = tag_begin + tag_end
        df.loc[:, 'buy'] = df['buy_tag'].ne('') #& tag_begin_end.ne('11 33')
        # df.loc[df['buy'], 'buy_tag'] = 'vws ' + tag_begin_end
        self.fill_custom_buy_info(df, metadata)
        # print(df.loc[df['buy'], ['date', 'close', 'hlc3_corr_10', 'lc2_adj', 'lc2_low_90']])
        return df

class ViNSellCorr(ViN):
    lookback_candles = 75
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1 = df_trade.iloc[-1]
        d = candle_1['date'].strftime('%Y-%m-%d %H:%M')
        ef = df_trade[['close', 'hlc3_adj', 'volume', 'lc2_adj', 'buy']].reset_index()
        buy_vol = ef['volume'].iat[0]
        trade_vol = ef['volume'].tail(trade_len).sum()
        trade_recent_buys = ef['buy'].tail(min(trade_len, 6)).sum()
        cp = (candle_1['close'] - trade.open_rate) / trade.open_rate
        if trade_len <= 2 or trade_recent_buys >= 1 or candle_1['streak_h'] >= candle_1['streak_s_min'] or (trade_vol < buy_vol * 0.1 * trade_len and -0.06 < cp < 0.02):
            return None
        t = 'profit' if cp >= 0.005 else 'loss'
        t += f" ({trade.buy_tag[:3]})"
        i = min(trade_len, self.lookback_candles - int(cp * 18))
        j = i // 2
        close_corr_i = ef['index'].rolling(window=i, min_periods=1).corr(ef['close'], method='spearman').iat[-1]
        close_corr_j = ef['index'].rolling(window=j, min_periods=1).corr(ef['close'], method='spearman').iat[-1]
        close_corr_ij_diff = close_corr_i - close_corr_j
        if cp < -0.04:
            if close_corr_ij_diff < -0.18 - 2 * cp and candle_1['streak_s_max'] < 1 and candle_1['lc2_adj'] / candle_1['close'] >= 0.999:
                log.info(f"{d} custom_sell: corr sell for pair {pair} with loss {cp:.2f} and trade len {trade_len}.")
                return f"corr {t}"
        elif cp > 0.015:
            if close_corr_ij_diff > 0.18 + 0.5 * cp:
                log.info(f"{d} custom_sell: corr sell for pair {pair} with profit {cp:.2f} and trade len {trade_len}.")
                return f"corr {t}"
        if trade_len > self.lookback_candles:
            close_min_j = ef['close'].tail(j).min()
            close_max_j = ef['close'].tail(j).max()
            if close_max_j / close_min_j < min(1.04, trade_len / self.lookback_candles) and candle_1['streak_s_max'] < 1 and candle_1['streak_s_min'] < 0:
                log.info(f"{d} custom_sell: sideways sell for pair {pair} with profit/loss {cp:.2f} and trade len {trade_len}.")
                return f"side close {t}"
        return None

class ViNSellRiseFall(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date)]
        trade_len = len(df_trade)
        candle_1 = df_trade.iloc[-1]
        trade_recent_buys = df_trade['buy'].tail(min(trade_len, 6)).sum()
        if trade_len <= 2 or trade_recent_buys >= 1 or candle_1['streak_s_min'] >= -1 or candle_1['streak_s_max'] >= 1:
            return None
        hlc3_min = df_trade['hlc3_adj'].tail(trade_len).min()
        hlc3_max = df_trade['hlc3_adj'].tail(trade_len).max()
        candle_min = df_trade.loc[df_trade['hlc3_adj'] <= 1.001 * hlc3_min].iloc[-1]
        candle_max = df_trade.loc[df_trade['hlc3_adj'] >= 0.999 * hlc3_max].iloc[-1]
        rise = candle_max['hlc3_adj'] / candle_min['hlc3_adj'] if candle_max['date'] > candle_min['date'] else 1
        fall = candle_max['hlc3_adj'] / candle_1['hlc3_adj']
        cp = (candle_1['close'] - trade.open_rate) / trade.open_rate
        t = 'profit' if cp >= 0.005 else 'loss'
        u = trade.buy_tag[:3]
        d = candle_1['date'].strftime('%Y-%m-%d %H:%M')
        if rise > 1.08:
            if fall > pow(rise, 0.33) - trade_len / 1000:
                log.info(f"{d} custom_sell: rise sell for pair {pair} with {t} {cp:.2f} and trade len {trade_len}.")
                return f"rise {u}"
        elif rise < 1.04:
            if fall > 1.16 - trade_len / 500:
                log.info(f"{d} custom_sell: fall sell for pair {pair} with {t} {cp:.2f} and trade len {trade_len}.")
                return f"fall {u}"
        j = 90
        if trade_len > j:
            hlc3_min = df_trade['hlc3_adj'].tail(j).min()
            hlc3_max = df_trade['hlc3_adj'].tail(j).max()
            if hlc3_max / hlc3_min < min(1.06, trade_len / j) and candle_1['streak_s_max'] < 1 and candle_1['streak_s_min'] < 0:
                log.info(f"{d} custom_sell: sideways sell for pair {pair} with {t} {cp:.2f} and trade len {trade_len}.")
                return f"side {u}"
        return None

class ViNSellRiseCorrFall(ViN):
    lookback_candles = 75
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        sell_reason = ViNSellCorr.custom_sell(self, pair, trade, current_time, current_rate, current_profit)
        if sell_reason is not None:
            return sell_reason
        sell_reason = ViNSellRiseFall.custom_sell(self, pair, trade, current_time, current_rate, current_profit)
        if sell_reason is not None:
            return sell_reason
        return None

class ViNSellEps(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date)]
        trade_len = len(df_trade)
        candle_1 = df_trade.iloc[-1]
        if trade_len <= 2:
            return None
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        d = candle_1['date'].strftime('%Y-%m-%d %H:%M')
        if current_profit < -0.18:
            log.info(f"{d} custom_sell: stop sell for pair {pair} with loss {current_profit:.2f} and trade len {trade_len}.")
            return f"stop"
        elif current_profit > 0.03:
            log.info(f"{d} custom_sell: profit sell for pair {pair} with profit {current_profit:.2f} and trade len {trade_len}.")
            return f"sell"
        elif trade_len > self.startup_candle_count:
            log.info(f"{d} custom_sell: sideways sell for pair {pair} with profit/loss {current_profit:.2f} and trade len {trade_len}.")
            return f"side"
        return None

class ViNPctCorr(ViNBuyPct, ViNSellCorr):
    pass

class ViNPctEps(ViNBuyPct, ViNSellEps):
    pass

class ViNPctRiseFall(ViNBuyPct, ViNSellRiseFall):
    pass

class ViNPctRiseCorrFall(ViNBuyPct, ViNSellRiseCorrFall):
    pass

class ViNVwsCorr(ViNBuyVws, ViNSellCorr):
    pass

class ViNVwsRiseFall(ViNBuyVws, ViNSellRiseFall):
    pass

class ViNVwsEps(ViNBuyVws, ViNSellEps):
    pass

class ViNLc2Corr(ViNBuyLc2, ViNSellCorr):
    pass

class ViNLc2Eps(ViNBuyLc2, ViNSellEps):
    pass

class ViNLc2RiseFall(ViNBuyLc2, ViNSellRiseFall):
    pass

class ViNPctLc2RiseFall(ViNBuyPctLc2, ViNSellRiseFall):
    pass

class ViresInNumeris(ViNBuyPct, ViNSellRiseFall):
    pass