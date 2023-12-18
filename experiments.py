import pandas as pd
from functools import lru_cache
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# if the symbol has no marketcap, there is no historical data for that year
np.set_printoptions(edgeitems=3000, linewidth=1000)
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 100000
# AAPL cash flow statement only goes back to 1989 while income statement
# goes back to 1985. so columns like commonstockissued will be nan and 
# we dont want to drop the whole row so first fill nan

def prepare_universe():
  df = pd.read_feather('master_screener.feather')
          
  #df[df.date==2000].isna().sum()        
          
  df.commonStockIssued.fillna(value=0, inplace=True)
  df.operatingCashFlow.fillna(value=0, inplace=True)
  df.totalAssets.fillna(value=0, inplace=True)

  df.roa.fillna(value=0, inplace=True)
  df.d_roa.fillna(value=0, inplace=True)
  df.d_lever.fillna(value=0, inplace=True)
  df.d_current.fillna(value=0, inplace=True)
  df.d_margin.fillna(value=0, inplace=True)
  df.d_asset.fillna(value=0, inplace=True)

  # only column with nans should be mve
  #df[df.date==2020].isna().sum()


  #len(df.dropna())
  # drop years where no marketcap exists
  df = df.dropna()

  #df[['symbol', 'date']].groupby('date').count()

  #df["bm_quintile"] = pd.qcut(df.bm, 5, labels=False)
  #df["mve_quintile"] = pd.qcut(df.mve, 3, labels=['small','med','large'])
  
  # rebalance yearly
  df['bm_quintile'] = df.groupby('date').bm.transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
  df['mve_quintile'] = df.groupby('date').mve.transform(
      lambda x: pd.qcut(x, 3, labels=False, duplicates='drop'))
  # df['mve_quintile'] = df.mve_quintile.astype(int).map({0:'low', 1:'med',2:'high'})
  
  pgroup = {
      0:"lo",
      1:"lo",
      2:"med",
      3:"med", 
      4:"med", 
      5:"med", 
      6:"med",
      7:"med",
      8:"hi",
      9:"hi"
  }

  df['pscore_group'] = df.pscore.astype(int).map(pgroup)
  #df = df[df.bm_quintile==4]

  return df



@lru_cache(maxsize=10000)
def get_hist(ticker):
  feather_hist = rel(f'hist_feather/{ticker}')
  return pd.read_feather(feather_hist)

# def hist_overlap_range(ticker, start, end):
  # hist = get_hist(ticker)
  # start = pd.to_datetime(start)
  # end = pd.to_datetime(end)
  # return (
            # (start <= hist.index[0] <= end) or
            # (start <= hist.index[-1] <= end) or
            # (hist.index[0] <= start <= hist.index[-1]) or
            # (hist.index[0] <= end <= hist.index[-1])
          # )

@lru_cache(maxsize=100000)
def computeIndividualReturn(ticker, start, end):
    
  df = get_hist(ticker)
  firstdate = pd.to_datetime(start)
  firstiloc = max(df.index.get_indexer([firstdate], method='ffill')[0], 0)
  lastdate = pd.to_datetime(end)
  lastiloc = max(df.index.get_indexer([lastdate], method='ffill')[0], 0)
  if firstiloc == lastiloc:
    return None
  first = df.iloc[firstiloc].adjClose.item()
  returns = (df.iloc[firstiloc:lastiloc].adjClose / first - 1.0).values
  #assert np.isnan(returns).sum() == 0  
  if returns[-1] > 300:
    print(ticker)
    asdf
  return returns

def computeReturns(portfolio, start, end):
  symbols = portfolio.symbol.tolist()
  returns = [computeIndividualReturn(ticker, start, end) for ticker in symbols]
  returns = [r for r in returns if r is not None]
  if len(returns) == 0:
    return None
  try:
    width = np.array([len(r) for r in returns]).max()
  except:
    print(len(r))
    asdf
  returns2 = np.zeros((len(returns), width))
  for i, r in enumerate(returns):
    returns2[i, width - len(r):] = r # align to end
    returns2[i, :width - len(r)] = r[0] # backfill front
  # equal weighted portolio returns
  eq_wght = returns2.mean(axis=0)
  
  return_vector = [eq_wght[-1], 
                    *np.percentile(eq_wght, [0, 10, 25, 50, 75, 90, 100]).tolist(), 
                    eq_wght.mean(), 
                    (1.0*(eq_wght > 0.0)).mean()
                  ]
  return return_vector
def get_piotroski_experiment_results(master, years = None):
  max_pfolio_size=10
  trials = 1000
  
  if years is None:
    years = master.date.unique()
    years.sort()
    
  filters = {
    # **{f'p{i}': f'pscore == {i*1.0}' for i in range(10)},
    'plo': 'pscore <= 1.0',
    'phi': 'pscore >= 8.0',
    'all': '~index.isnull()',# no filter
    'ps_lo': '~index.isnull()',
    'ps_hi': '~index.isnull()',
  }
  rows = []
  for year in years:#[2022]:
    
    basket = master[master.date==year]
    start = f'{year}-01-01'
    end = f'{year+1}-01-01'
    for filtername, filter in filters.items():
      print(year, filtername)
      df = basket.query(filter)
      pfolio_size = min(len(df), max_pfolio_size)
      if not pfolio_size:
        print(year, len(df), 'cannot run experiment')
        continue
      results = [computeReturns(df.sample(pfolio_size), start, end) for _ in range(trials)]
      results = [r for r in results if r is not None] 
      rows.extend([[start, end, filtername, *r, len(df) ] for r in results])
  final = pd.DataFrame(columns=['start', 'end', 'group', 'returns', 'p0', 'p10', 'p25', 'p50', 'p75', 'p90', 'max', 'mean', 'positive', 'n'], data=rows)
  return final

 
def run_piotroski_tests(df):
  starts = df.start.unique()
  starts.sort()
  #print(starts)
  data = []
  hilos = []
  hialls = []
  ps_hilos = []
  for start in starts:#['2020-01-01']:
    hi = df[(df.start == start) & (df.group == 'phi')]
    lo = df[(df.start == start) & (df.group == 'plo')]
    all = df[(df.start == start) & (df.group == 'all')]
    ps_hi = df[(df.start == start) & (df.group == 'ps_hi')]
    ps_lo = df[(df.start == start) & (df.group == 'ps_lo')]
    #print(start)
    _min = np.array([len(hi), len(lo), len(all), len(ps_hi), len(ps_lo)]).min()
    if _min == 0:
      #print('cannot test', start, len(hi), len(lo), len(all), len(ps_hi), len(ps_lo))
      continue
    hilo = hi.returns.values[:_min] - lo.returns.values[:_min]
    ps_hilo = ps_hi.returns.values[:_min] - ps_lo.returns.values[:_min]
    hiall = hi.returns.values[:_min] - all.returns.values[:_min]
    hilos.append(hilo[:20])
    ps_hilos.append(ps_hilo[:20])
    hialls.append(hiall[:20])
    t_hilo = stats.ttest_ind(hilo, ps_hilo)
    t_hiall = stats.ttest_ind(hi.returns.values[:_min] - all.returns.values[:_min], ps_hi.returns.values[:_min] - ps_lo.returns.values[:_min])
    data.append([start, end, 'high - low', _min, hilo.mean(), ps_hilo.mean(), t_hilo.statistic, t_hilo.pvalue])
    data.append([start, end, 'high - all', _min, hiall.mean(), ps_hilo.mean(), t_hiall.statistic, t_hiall.pvalue])
  hilos = np.concatenate(hilos)
  hialls = np.concatenate(hialls)
  ps_hilos = np.concatenate(ps_hilos)
  print(hilos.shape)
  
  t_hilos = stats.ttest_ind(hilos, ps_hilos)
  data.append(['--', '--', 'high - low', len(hilos), hilos.mean(), ps_hilos.mean(), t_hilos.statistic, t_hilos.pvalue])
  t_hialls = stats.ttest_ind(hialls, ps_hilos)
  data.append(['--', '--', 'high - all', len(hilos), hialls.mean(), ps_hilos.mean(), t_hialls.statistic, t_hialls.pvalue])
    
    
  return pd.DataFrame(columns=['start', 'end', 'test', 'n', 'mu1', 'mu2', 't-statistic', 'p-value'], data=data)

def bull_bear(master):
  df = get_hist('^GSPC')[['adjClose']]
  print(df.index)
  df['dd'] = df.adjClose.div(df.adjClose.cummax()).sub(1)
  df['ddn'] = ((df['dd'] < 0.) & (df['dd'].shift() == 0.)).cumsum()
  df['ddmax'] = df.groupby('ddn')['dd'].transform('min')
  df['bear'] = (df['ddmax'] < -0.2) & (df['ddmax'] < df.groupby('ddn')['dd'].transform('cummin'))
  df['bearn'] = ((df['bear'] == True) & (df['bear'].shift() == False)).cumsum()

  bears = df.reset_index().query('bear == True').groupby('bearn').date.agg(['min', 'max'])
  print(bears)
  bulls = df.reset_index().query('bear == False').groupby('bearn').date.agg(['min', 'max'])
  print(bulls)
  
  
  # df.adjClose.plot()

  # for i, row in bears.iterrows():
      # plt.fill_between(row, df.adjClose.max(), alpha=0.25, color='r')
  # plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
  # plt.ylabel('S&P 500 Index (^GSPC)')
  # plt.title('S&P 500 Index with Bear Markets (> 20% Declines)')

  # plt.savefig('bears.png')
  # plt.show()
  max_pfolio_size=10
  trials = 1000
  years = master.date.unique()
  years.sort()
  types = [bears, bulls]
  typenames = ['bear', 'bull']
  results = {}
  
  for year in years:
    basket = master[master.date==year]
    phi = basket[basket.pscore >= 8.0]
    for i, type in enumerate(types):
      typeset = type[pd.DatetimeIndex(type['min']).year == year]
      #print(year, typenames[i], typeset)
      for _, row in typeset.iterrows():
        start = row["min"].strftime('%Y-%m-%d')
        end = row["max"].strftime('%Y-%m-%d')
        print(typenames[i], start, end)
        pfolio_size = min(len(phi), max_pfolio_size)
        if not pfolio_size:
          print(year, len(phi), 'cannot run experiment')
          continue
        print(start, end, 'pfolio_size', pfolio_size, 'phi pool size', len(phi))
        hi = np.stack([computeReturns(phi.sample(pfolio_size), start, end) for _ in range(trials)])
        al = np.stack([computeReturns(basket.sample(pfolio_size), start, end) for _ in range(trials)])
    
        print('hi', np.nanmean(hi, axis=0))
        #if np.isnan(al).sum() > 0:
        #  print(al)
        #  asdf
        print('al', np.nanmean(al, axis=0))
        
      
def panelA(master):
  for pscore in np.arange(0., 9., 1.):
    basket = master[master.pscore == pscore]
    print(basket.pscore)
    
    
if __name__ == '__main__':
  #duplicate_piotroski_tests(master)
  #bull_bear(master)
  panelA(master)