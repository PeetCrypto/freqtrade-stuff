"""
Binance Marketcap PairList provider
Provides dynamic pair list based on market cap
"""
import logging
import requests
from typing import Any, Dict, List

from cachetools.ttl import TTLCache

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class BinanceMarketCapPairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')

        self._stake_currency = config['stake_currency']
        self._number_pairs = self._pairlistconfig['number_assets']
        self._refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._url = \
            'https://www.binance.com/exchange-api/v2/public/asset-service/product/get-products'

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist. '
                'Please edit your config and restart the bot.'
            )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._pairlistconfig['number_assets']} marketcap pairs."

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """
        # Generate dynamic whitelist
        # Must always run if this pairlist is not the first in the list.
        pairlist = self._pair_cache.get('pairlist')
        if not pairlist:
            resp = requests.get(self._url)
            data = resp.json()
            pairlist = []
            for i in data["data"]:
                if (i["s"].endswith(self._stake_currency)
                        and i["cs"] is not None and i["c"] is not None):
                    capital = i["cs"] * float(i["c"])
                    pair = i["s"].replace(self._stake_currency, f"/{self._stake_currency}")
                    pairlist.append([pair, capital])

            pairlist.sort(key=lambda tup: tup[1], reverse=True)
            pairlist = [tup[0] for tup in pairlist]
            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache['pairlist'] = pairlist

        return pairlist

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Use the incoming pairlist.
        pairlist = [k for k in tickers.keys() if k in pairlist]
        # Validate whitelist to only have active market pairs
        pairlist = self._whitelist_for_active_markets(pairlist)
        pairlist = self.verify_blacklist(pairlist, logger.info)
        # Limit pairlist to the requested number of pairs
        pairlist = pairlist[:self._number_pairs]
        self.log_once(f"Searching {self._number_pairs} pairs: {pairlist}", logger.info)

        return pairlist