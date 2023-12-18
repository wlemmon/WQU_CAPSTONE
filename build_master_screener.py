from utils import *

tickers = get_tickers()
dfs = []
for ticker in tickers:  
  print(ticker)
  feather_fund = rel(f'fund_feather/{ticker}')
  if os.path.exists( feather_fund) and os.path.getsize(feather_fund) > 0:
    df = pd.read_feather(feather_fund)
    df.index = pd.PeriodIndex( df.index, freq='Y')   
    df = df.reset_index()
    df['symbol'] = ticker
    dfs.append(df)
    
df = pd.concat(dfs)
df.date = df.date.astype(str).astype(int)

df.to_feather('master_screener.feather')