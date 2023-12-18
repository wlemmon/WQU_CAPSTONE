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
tickers = get_tickers()
    
def worker(i, ticker, session, overwrite_final=False):
  # print('ticker')  
  
  csv_fund = rel(f'fund_csv/{ticker}')
  feather_fund = rel(f'fund_feather/{ticker}')
  feather_hist = rel(f'hist_feather/{ticker}')
  if not os.path.exists( feather_fund) or overwrite_final:
  # if os.path.getsize(feather_fund) != 0:
    # fund = pd.read_feather(feather_fund)
    # if (fund.d_current == np.inf).sum() == 0:
     # return
    # print(ticker)
    
    if not os.path.exists(feather_hist):
      print('hist file not present', ticker)
      return
    if os.path.getsize(feather_hist) == 0:
      touch(feather_fund)
      return
    bs = balance_sheet(ticker, session=session)
    if type(bs) is dict:
      print('error bs', ticker, bs)
      time.sleep(10)
      return
    assert type(bs) == list
    if len(bs) == 0:
      print('empty bs', ticker, bs)
      touch(feather_fund)
      return
    iis = income_statement(ticker, session=session)  
    if type(iis) is dict:
      print('error iis', ticker, iis)
      time.sleep(10)
      return
    assert type(bs) == list
    if len(iis) == 0:
      print('empty iis', ticker, iis)
      touch(feather_fund)
      return
    cf = cashflow_statement(ticker, session=session)
    if type(cf) is dict:
      print('error cs', ticker, cf)
      time.sleep(10)
      return
    assert type(bs) == list
    if len(cf) == 0:
      print('empty cs', ticker, cf)
      touch(feather_fund)
      return
    
    bs = pd.DataFrame(bs)
    bs = bs.set_index('date').sort_index()
    bs = bs[~bs.index.duplicated(keep='first')]
    iis = pd.DataFrame(iis)
    iis = iis.set_index('date').sort_index()
    iis = iis[~iis.index.duplicated(keep='first')]
    cf = pd.DataFrame(cf)
    cf = cf.set_index('date').sort_index()
    cf = cf[~cf.index.duplicated(keep='first')]
    hist = pd.read_feather(feather_hist)
    if 'fillingDate' not in bs:
      print('fillingDate not in bs', ticker)
      time.sleep(15)
      return
    if 'fillingDate' not in iis:
      print('fillingDate not in iis', ticker)
      time.sleep(15)
      return
    if 'fillingDate' not in cf:
      print('fillingDate not in cf', ticker)
      time.sleep(15)
      return
    
    cols_to_use = iis.columns.difference(bs.columns)
    comb = pd.merge(bs, iis[cols_to_use], left_index=True, right_index=True, how='inner')
    cols_to_use = cf.columns.difference(comb.columns)
    comb = pd.merge(comb, cf[cols_to_use], left_index=True, right_index=True, how='inner')
    # convert tim datetimeindex for mergability with hist data
    comb.index = pd.DatetimeIndex(comb.index)
    comb = piotroski_score(comb, hist)
    # drop rows where price is NaN
    comb = comb[~comb.price.isna()]
    
    columns_of_interest = [
      'totalAssets',
      'mve', 
      'commonStockIssued',
      "roa",
      "operatingCashFlow",
      "book",
      "price",
      "bm",
      "d_roa",
      "d_lever",
      "d_current",
      "d_margin",
      "d_asset",
      "i_roa",
      "cf",
      "i_cf",
      "id_roa",
      "i_accrual",
      "id_lever",
      "id_current",
      "i_shares",
      "id_margin",
      "id_asset",
      'pscore'
    ]
    comb = comb[columns_of_interest]
    
         
    comb = comb.astype('float32')
    
    if i % 100 == 0 or ticker in ['AAPL', 'GOOG']:
      comb.to_csv(csv_fund, sep=',', index=True, encoding='utf-8')
    comb.to_feather(feather_fund)
    print('finished write for', ticker)
  else:
    pass
    # print('skipping already combined', ticker)
  return
# tickers = ['AAIC']
# tickers = tickers[:1000]
max_workers=2
sessions = [requests.session() for x in range(max_workers)]
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

  future_to_ticker = {executor.submit(worker, i, ticker, sessions[i%max_workers]): ticker for i, ticker in enumerate(tickers)}
  for future in concurrent.futures.as_completed(future_to_ticker):
      ticker = future_to_ticker[future]
      try:
        data = future.result()
      except Exception as exc:
          print('%r generated an exception: %s' % (ticker, exc))
          print(traceback.format_exc())

              
            