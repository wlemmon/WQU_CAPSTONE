from datetime import datetime
import financedatabase as fd
from financetoolkit import Toolkit

import os

import certifi
api_key='kxTL6LihZ6T7cCWQwyfrkJFxlJOvrSPC'

start = datetime(2011, 1, 1)
end = datetime(2012, 12, 31)
project_root = 'D:\WQU_CAPSTONE'
def rel(dir):
    return os.path.join(project_root, dir)
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)

from io import StringIO
import requests
try:
  # For Python 3.0 and later
  from urllib.request import urlopen
except ImportError:
  # Fall back to Python 2's urllib2
  from urllib2 import urlopen

import os
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)
def sanitize_ticker_name(ticker):
  return ticker.replace('/', '-')
   
def get_tickers():

  # Initialize the Equities database
  base_url='D:/WQU_CAPSTONE/FinanceDatabase-main/compression/'
  equities = fd.Equities(base_url=base_url, use_local_location=False)
  tkus = equities.search(market=['NASDAQ Capital Market', 'NASDAQ Global Select', 'New York Stock Exchange'])
  _tickers = tkus.index.tolist()
  # remove some problematic tickers
  _tickers.remove('NVCN')
  _tickers.remove('VBIV')
  tickers = []
  for t in _tickers:
    if type(t) == str:
      tickers.append(sanitize_ticker_name(t))
  return tickers
def get(ticker, type, session=None, fromdate='1980-01-01', todate=None):  
  if session is None:
    session = requests.session()
  tostr = f"&to={todate}" if todate else ""
  url = (f"https://financialmodelingprep.com/api/v3/{type}/{ticker}?period=year&apikey={api_key}&from={fromdate}{tostr}")
  response = session.get(url)#, timeout=60)
  #response = urlopen(url, cafile=certifi.where())
  return response.json()

def balance_sheet(ticker, session=None):
  return get(ticker, 'balance-sheet-statement', session=session)

def income_statement(ticker, session=None):
  return get(ticker, 'income-statement', session=session)

def cashflow_statement(ticker, session=None):
  return get(ticker, 'cash-flow-statement', session=session)
  
def historical(ticker, session=None):
  return get(ticker, 'historical-price-full', session=session)
  
def marketcap(ticker, session=None):
  partials = []
  stride = 5
  for year in range(2005, 2024, stride):
    print('marketcap', ticker, year)
    partials.append(pd.DataFrame(get(
        ticker, 
        'historical-market-capitalization', 
        session=session,
        fromdate=f'{year}-01-01',
        todate=f'{year+stride}-01-01'
      )))
    
  full = pd.concat(partials)
  if 'date' in full:
    full = full.set_index('date').sort_index()
  return full
  
def clean_adj_close(df):
    df = df['Adj Close']
    # print(np.where(cols.max() > 1000000000))
    todrop=df.columns[np.where(df.max() > 1000000000)]
    return df.drop(columns=todrop, axis=1)
 

def piotroski_score(comb, hist):
  net_income = comb['netIncome']
  # don't allow negative assets
  # to prevent div by zero
  comb['totalAssets'] = comb['totalAssets'].clip(lower = 0.01)
  comb['revenue'] = comb['revenue'].clip(lower = 0.01)
  comb['totalCurrentLiabilities'] = comb['totalCurrentLiabilities'].clip(lower = 0.01)
  # total shares outstanding
  comb["weightedAverageShsOutDil"] = comb["weightedAverageShsOutDil"].clip(lower = 0.01)
  
  total_assets_begin = comb['totalAssets'].shift().bfill()
  total_assets_end = comb['totalAssets']
  share_price = pd.merge(hist, comb['totalAssets'], left_index=True, right_index=True, how='outer').ffill().loc[comb.index, 'adjClose']
  
  comb['book'] = (comb['totalStockholdersEquity'] - comb['preferredStock']).clip(lower=0.0) / comb["weightedAverageShsOutDil"]
  # don't allow negative book value
  comb['book'] = comb['book'].clip(lower = 0.01)
  
  comb['price'] = share_price
  comb['bm'] = comb.book / share_price
 
  
  comb["mve"] = share_price * comb["weightedAverageShsOutDil"] 
  comb["roa"] = net_income / ((total_assets_begin + total_assets_end) / 2).clip(lower = 0.01)
  
  comb["i_roa"] = 1.0*(comb.roa > 0)

  comb["i_cf"] = 1.0*(comb['operatingCashFlow'] > 0)

  comb["d_roa"] = comb.roa.diff().fillna(value=0)
  comb["id_roa"] = 1.0*(comb.d_roa > 0)

  operating_cf_to_total_assets = comb['operatingCashFlow'] / comb['totalAssets']
  comb["cf"] = operating_cf_to_total_assets
  comb["i_accrual"] = 1.0*(comb.cf > comb.roa)

  debt_ratio = comb['totalDebt'] / comb['totalAssets']
  comb["lever"] = debt_ratio
  comb["d_lever"] = debt_ratio.diff().fillna(value=0)
  comb["id_lever"] = 1.0 * (comb.d_lever > 0)

  current_ratio = comb['totalCurrentAssets'] / comb['totalCurrentLiabilities']
  comb["current"] = current_ratio
  comb["d_current"] = current_ratio.diff().fillna(value=0)
  comb["id_current"] = 1.0 * (comb.d_current > 0)

  comb["i_shares"] = 1.0*(comb['commonStockIssued'] == 0)

  gross_margin = (comb['revenue'] - comb['costOfRevenue']) / comb['revenue']
  comb["margin"] = gross_margin
  comb["d_margin"] = gross_margin.diff().fillna(value=0)
  comb["id_margin"] = 1.0 * (comb.d_margin > 0)

  asset_turnover_ratio = comb['revenue'] / ((total_assets_begin + total_assets_end) / 2)
  comb["d_asset"] = asset_turnover_ratio.diff().fillna(value=0)
  comb["id_asset"] = 1.0 * (comb.d_asset> 0)

  comb["pscore"] = (
      comb.i_roa+
      comb.i_cf+
      comb.id_asset+
      comb.i_accrual+
      comb.id_lever+
      comb.id_current+
      comb.i_shares+
      comb.id_margin+
      comb.id_asset
  )
  return comb
  
  
 
if __name__ == '__main__':
  marketcap('AAPL')