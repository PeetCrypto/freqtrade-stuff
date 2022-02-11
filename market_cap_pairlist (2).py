import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from freqtrade.configuration import Configuration
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.misc import chunks
from freqtrade.resolvers import StrategyResolver
from pycoingecko import CoinGeckoAPI

logger = logging.getLogger(__name__)


class MarketCapPairList:
    def __init__(self, coins_list, currency) -> None:

        self._stake_currency = currency
        self._cg = CoinGeckoAPI()
        self._marketcap_ranks: Dict[str, Any] = {}
        self._coins_list = coins_list


    def _request_marketcap_ranks(self):

        _marketcap_ranks = {}

        chunks_no = 50

        # seems coingecko limits the number of ids to 52, using 50.
        for _ids in chunks(self._coins_list, chunks_no):

            base_marketkaps = self._cg.get_coins_markets(
                    vs_currency='usd',
                    symbols=','.join(_ids).lower(),
                    order='market_cap_desc',
                    per_page=len(self._coins_list),
                    sparkline=False,
                    page=1)
            for x in base_marketkaps:
                symbol = x['symbol'].upper()
                rank = x['market_cap_rank']
                if rank and symbol not in _marketcap_ranks:
                    _marketcap_ranks[symbol] = rank

        self._marketcap_ranks = _marketcap_ranks

    def gen_pairlist(self) -> List[str]:

        self._request_marketcap_ranks()

        # sort marketcaps
        pairlist = [
                k.upper() + f'/{self._stake_currency}'
                for k, v in sorted(self._marketcap_ranks.items(), key=lambda x: x[1])
                ]

        return pairlist

def main():
    parser = argparse.ArgumentParser(description='Print the top trading pairs by market cap.')
    parser.add_argument('-n', type=int, dest='no_of_pairs', default=60, action='store', help='Number of pairs to print.')
    parser.add_argument('--stake_currency', type=str, dest='stake_currency', action='store', required=True, help='Stake currency.')
    parser.add_argument('--blacklist', type=str, dest='blacklist', action='store', required=True, help='Coin blacklist.')

    args = parser.parse_args()

    assert Path(args.blacklist).is_file(), "Invalid blacklist file path."

    freqtrade_config = Configuration.from_files([args.blacklist])
    freqtrade_config['stake_currency'] = args.stake_currency.upper()

    all_pairs_sorted_by_volume = {'method': 'VolumePairList', 'number_assets': 99999, 'sort_key': 'quoteVolume'}
    freqtrade_config['pairlists'] = [all_pairs_sorted_by_volume]
    freqtrade_config['dry_run'] = True
    freqtrade_config['timeframe'] = "5m"

    if 'db_url' not in freqtrade_config:
        freqtrade_config['db_url'] = 'sqlite://'

    mock_strategy = MagicMock()
    mock_strategy.timeframe = "5m"

    with patch.object(StrategyResolver, 'load_strategy', return_value=mock_strategy):  # This is greasy
        freqtrade = FreqtradeBot(freqtrade_config)

    total_available_pairs = freqtrade.pairlists.whitelist

    symbols = [symbol.split("/")[0] for symbol in total_available_pairs]

    market_cap_pairs = MarketCapPairList(symbols, freqtrade.config['stake_currency'])
    pair_list = market_cap_pairs.gen_pairlist()[:args.no_of_pairs]
    template = { "exchange": {
                        "name": freqtrade_config['exchange']['name'],
                        "pair_whitelist": pair_list
                    },
                    "pairlists": [
                        {
                            "method": "StaticPairList"
                        }
                    ]
                }
    print(json.dumps(template, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
