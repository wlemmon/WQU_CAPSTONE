api_key='kxTL6LihZ6T7cCWQwyfrkJFxlJOvrSPC'
import time
from utils import *
import financedatabase as fd
import traceback
import threading
import json
from io import StringIO
import concurrent.futures
import urllib.request
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 100000

tickers = get_tickers()
tickers.append('^GSPC')

def worker(i, ticker, session, overwrite_final=False):
  print('ticker')
  
  csv_fund = rel(f'fund_csv/{ticker}')
  feather_fund = rel(f'fund_feather/{ticker}')
  csv_hist = rel(f'hist_csv/{ticker}')
  feather_hist = rel(f'hist_feather/{ticker}')
  if not os.path.exists( feather_hist) or overwrite_final:
    print('downloading historical', ticker)
    historical_data = historical(ticker, session=session)
    if len(historical_data):
      if "historical" not in historical_data:
        print(historical_data)
        time.sleep(10)
        return
      historical_data = pd.DataFrame(historical_data["historical"])
      #historical_data = historical_data[['date', 'open', 'high', 'low', 'close', 'adjClose', 'volume']]
      # sometimes adjClose has nan values in this data source (happens for only 1/20_000 tickers: GDL-PC). for those
      # tickers, use close price instead
      if historical_data.adjClose.isna().sum() > 0:
        print('adjClose has nans', historical_data.adjClose.isna().sum())
        print('close nans', historical_data.close.isna().sum())
        historical_data['adjClose'] = historical_data['close']
      # sometimes the first day of an IPO shows up as 0.00001 and should be removed to prevent
      # spuriously huge returns. if this only happens once per hist, assume its an ipo entry.
      # sometimes several of the first entries are this way. lets just remove all of them even
      # if they are not in the front.
      if ((historical_data.adjClose <= 0.00001) & (historical_data.volume == 0)).sum():
        historical_data = historical_data[~((historical_data.adjClose <= 0.00001) & (historical_data.volume == 0))]
      historical_data = historical_data[['date', 'adjClose', 'volume']]
      #historical_data = historical_data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume":
      # "Volume"})
      historical_data = historical_data.set_index('date').sort_index()
    
      historical_data = historical_data.astype('float32')
      historical_data.index = pd.DatetimeIndex(historical_data.index)
      if i % 100 == 0 or ticker in ['AAPL', 'GOOG']:
        historical_data.to_csv(csv_hist, sep=',', index=True, encoding='utf-8')
      historical_data.to_feather(feather_hist)
      print('finished write for', ticker)
    else:
      touch(feather_hist)
    
  else:
    print('skipping already combined', ticker)
  return
# tickers = ['GDL-PC']
# tickers = tickers[:20]
max_workers=3
import requests
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

              
            