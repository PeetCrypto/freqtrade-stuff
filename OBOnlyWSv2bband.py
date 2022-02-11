import numpy as np
import talib.abstract as ta
from datetime import datetime, timedelta
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime,timedelta
import math
from freqtrade.strategy.interface import IStrategy, SellCheckTuple, SellType
from talipp.indicators import EMA, SMA ,BB


from user_data.strategies.BinanceStream import BaseIndicator, OrderBook,BinanceStream


class OBOnlyWSv2bband(BinanceStream):
    INTERFACE_VERSION = 2


    stoploss = -0.11  


    timeframe = '1m'
    use_sell_signal = True
    sell_profit_only = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    strat_data={
        "ratio_buy1":False,
        "ratio_buy2":False,
        "ratio_buy3":0,
        "ratio_wall":0,
        "price":0,
        "ratio_ema":0,
        "price_ub":math.nan,
        "price_lb":math.nan,
        "ratio_gain":0,
    }
    def init_pair_info(self,pi):
  
        pi.buy_signal=0
        pi.bi=BaseIndicator(pi.pair, currency="USDT")
        pi.ob_bb=BB(200,2.0)
        pi.bb5=BB(20,2.0,input_indicator=pi.bi.c)
        pi.ob_ema=EMA(7)
        pi.sell_signal=0
        pi.ob=OrderBook(pi.pair,currency="USDT")
   
    def ob_cut(self, bids, asks,delta_bid,delta_ask=None,bid_weight=0.5):
        if delta_ask is None:
            delta_ask=delta_bid
        mid_price=(bid_weight*bids[0][0]+(1-bid_weight)*asks[0][0])
        bid_cut = mid_price - mid_price*delta_bid
        ask_cut = mid_price + mid_price*delta_ask
        bid_side=bids[bids[:,0]>bid_cut]
        ask_side=asks[asks[:,0]<ask_cut] 
        return bid_side,ask_side  
    def check_ob(self,pi , bids, asks,delta_bid,delta_ask=None,wall=0.0,ratio=1.0,bid_weight=0.5,reciprocal=False):
        if delta_ask is None:
            delta_ask=delta_bid
        mid_price=(bid_weight*bids[0][0]+(1-bid_weight)*asks[0][0])
        bid_cut = mid_price - mid_price*delta_bid
        ask_cut = mid_price + mid_price*delta_ask
        bid_side=bids[bids[:,0]>bid_cut]
        ask_side=asks[asks[:,0]<ask_cut]
        wall_side=bid_side
        asum=ask_side[:,1].sum() 
        bsum=bid_side[:,1].sum() 
        if wall<0:
            wall_side=ask_side
            wall=-wall
        wsum=wall_side[:,1].sum()
        r=bsum/asum 
        r_test=(r >ratio)
        if reciprocal:
            r_test= ((1/r) >ratio)
        if r_test and min(np.size(ask_side[:,1]),np.size(bid_side[:,1])) > 10:
            wlist=wall_side[wall_side[:,1]>(wall*wsum)]
            if   len (wlist) >0 :
                return True,r
        return False,r
    def rescale(self,r):
        if math.isnan(r) or math.isinf(r) or r==0:
            return 1
        if r>1:
            return r-1
        return -(1/r-1)   
    def process_ob(self,pi, bids, asks):
  
        bb=pi.ob_bb
        ema=pi.ob_ema
        self.strat_data["price"]=mid_price=(1*bids[0][0]+1*asks[0][0])/2

        _, r2=self.check_ob(pi,bids, asks,delta_bid=0.002,delta_ask=0.002,wall=0.4,ratio=1.7)
       
        bid_side,ask_side=self.ob_cut( bids, asks,delta_bid=0.002)
        mid_price=(1*bids[0][0]+1*asks[0][0])/2
  

        no_wallb=bid_side[bid_side[:,1]<0.4*np.sum(bid_side[:,1])]
        no_walla=ask_side[ask_side[:,1]<0.4*np.sum(ask_side[:,1])]

        r2=np.sum(bid_side)/np.sum(ask_side)
        r2=self.rescale(r2)  
        r2nw=np.sum(no_wallb)/np.sum(no_walla)
        r2nw=self.rescale(r2nw)  

        
        if len(bb)>0:      
            iv=r2nw
           
            #print(f"will added {iv} {bb[-1].lb}")
            bb.add_input_value(iv)
            #print(f" added {iv} {bb[-1].lb}")

            bb.purge_oldest(1)
            #print(f" pop {iv} {bb[-1].lb}")

           

           
        else:
            bb.add_input_value(r2nw)   
        if len(ema)>0:      
            self.strat_data["ratio_ema"]=ema[-1]
            
            ema.add_input_value(r2)
            ema.purge_oldest(1)
            
        else:
             ema.add_input_value(r2)

    def new_ob(self, pi, depth_cache):
        bids = np.array(depth_cache.get_bids())
        asks = np.array(depth_cache.get_asks())
        self.process_ob(pi,bids,asks)
        self.check_sell(pi,bids,asks)
        self.check_buy(pi,bids,asks)

    def check_buy(self,pi,bids, asks ):

        prev_buy_signal=pi.buy_signal
        pi.buy_signal=0
        pair=pi.pair

        open_trades= pi.open_trades()
        
        #### NO RETURN BEFORE HERE
        
        
       # if len (open_trades) >= 1 or self.no_trade_until > datetime.now():
       #     return
        mid_price=(1*bids[0][0]+1*asks[0][0])/2
        
        
        buy_price=(0.2*bids[0][0]+0.8*asks[0][0])

        if len(pi.bi.c) == 0 or pi.bi.c[-1][-1] > bids[0][0]:
             return
        bb5=pi.bb5
        if len(bb5)>0: 
            bbb=bb5[-1]
            cond1=mid_price<(bbb.cb)
            cond2 = (bbb.cb-bbb.lb)> mid_price * 0.004
            self.strat_data["ratio_buy1"]=cond1
            self.strat_data["ratio_buy2"]=cond2
            if not (cond1 and cond2):
                
                return 
        else:
            return           
 
        buy3=False
        bb=pi.ob_bb
        ema=pi.ob_ema
        if len(bb)>0 and len(ema)>0:      
            if ema[-1] > 1.2*bb[-1].ub:
                buy3=True
            
        self.strat_data["ratio_buy3"]= 1 if buy3 else 0

        if   buy3 : 
           pi.buy(buy_price)
     
    def check_sell(self, pi, bids, asks):
        
        sell_price=(0.1*bids[0][0]+0.9*asks[0][0])
        ob_price=(0.2*bids[0][0]+0.8*asks[0][0])
        mid_price=(0.5*bids[0][0]+0.5*asks[0][0])
        pair=pi.pair
        found_trade= pi.open_trades(pair=pair)
        prev_sell_signal=pi.sell_signal
        pi.sell_signal=0
        if(found_trade == None):
            return
        found_trade= pi.open_trades(force=True,pair=pair)
        if(found_trade == None):
            return
        print("found trade")
        if len(pi.bi.c) == 0 or  pi.bi.c[-1][-1] < asks[0][0]:
            return
        print("check1")
                
        gain = (mid_price-found_trade.open_rate)/found_trade.open_rate
        
        self.strat_data["ratio_gain"]= gain*100

        
        
        sell_1=False
        bb=pi.ob_bb
        ema=pi.ob_ema
        sell2,r2=self.check_ob(pair,bids=bids, asks=asks,delta_bid=0.002,delta_ask=0.002,ratio=1.,bid_weight=0.2,wall=-0,reciprocal=True)
        sell2=False 
        if r2 <1.0:
            sell2=True
        elapsed=datetime.now()-found_trade.open_date  
        elapsed_min=elapsed.total_seconds()//60
        elapsed_min2=max(0,elapsed_min-20)
        factor=max(0.8,1-elapsed_min2*0.005)    
        if len(bb)>0 and len(ema)>0:  
  
            if ema[-1] < 1*factor*bb[-1].lb:
                sell_1=True
        
        sell=False
        if sell_1 and sell2:
            pi.sell_signal=prev_sell_signal+1
           
 
            if gain > 0 or elapsed > timedelta(hours=24):
                pi.sell(asks[0][0])
  

        if gain >0.003 or  sell:
            pi.sell(sell_price)
       
            
