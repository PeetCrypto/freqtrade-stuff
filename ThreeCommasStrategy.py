import os
import logging
from freqtrade.persistence import Trade, PairLocks
from datetime import datetime, timedelta, timezone
from freqtrade.strategy.interface import IStrategy
from dotenv import load_dotenv
from py3cw.request import Py3CW

load_dotenv()

log = logging.getLogger(__name__)


# credits to https://github.com/AlexBabescu


class ThreeCommasStrategy(IStrategy):

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        if hasattr(super(), 'confirm_trade_entry'):
            raise NotImplementedError('confirm_trade_entry is already implemented, 3commas will not work.')

        coin, currency = pair.split('/')

        p3cw = Py3CW(
            key=os.getenv('P3CW_KEY'),
            secret=os.getenv('P3CW_SECRET'),
        )

        # action_id comes from params
        error, data = p3cw.request(
            entity='bots',
            action='start_new_deal',
            action_id=str(os.getenv('P3CW_BOT_ID')),
            payload={
                "bot_id": os.getenv('P3CW_BOT_ID'),
                "pair": f"{currency}_{coin}",
            },
        )

        log.info(f"{error} {data}")

        PairLocks.lock_pair(
            pair=pair,
            until=datetime.now(timezone.utc) + timedelta(minutes=5),
            reason="Send 3c buy order"
        )

        return False  # we don't want to keep the trade in freqtrade db
