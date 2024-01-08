api_key='kxTL6LihZ6T7cCWQwyfrkJFxlJOvrSPC'
import time
from utils import *
import traceback
import threading
import requests
import json
from io import StringIO
import concurrent.futures
import urllib.request
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 100000

# this file is to apply adhoc patches to saved feather files

tickers = get_tickers()

# for ticker in tickers:
  # hist_feather = rel(f'hist_feather/{ticker}')
  # if os.path.exists( hist_feather) and os.path.getsize(hist_feather) != 0:
    # hist = pd.read_feather(hist_feather)
    
    # if ((hist.adjClose <= 0.00001) & (hist.volume == 0)).sum():
      # print(ticker)
      # hist = hist[~((hist.adjClose <= 0.00001) & (hist.volume == 0))]
      # hist.to_feather(hist_feather)
for ticker in tickers:
  print(ticker)
  fund_feather = rel(f'fund_feather/{ticker}')
  if os.path.exists( fund_feather) and os.path.getsize(fund_feather) != 0:
    fund = pd.read_feather(fund_feather)
    #fund.i_shares = 1.0 - fund.i_shares
    fund.pscore = (
      fund.i_roa+
      fund.i_cf+
      fund.id_asset+
      fund.i_accrual+
      fund.id_lever+
      fund.id_current+
      fund.i_shares+
      fund.id_margin+
      fund.id_asset
  )
    fund.to_feather(fund_feather)
    