import pandas as pd
from functools import lru_cache
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

MARKET_TICKER = '^GSPC'
return_vector_field_names = ['raw_ret', 'raw_ret_pos', 'raw_avg_ret', 'raw_avg_ret_pos', 'p0', 'p10', 'p25', 'p50', 'p75', 'p90', 'max', 'ma_ret', 'ma_ret_pos', 'ma_avg_ret', 'ma_avg_ret_pos', 'm_ret']


def p5(g): return np.percentile(g, 5)
def p10(g): return np.percentile(g, 10)
def p25(g): return np.percentile(g, 25)
def p50(g): return np.percentile(g, 50)
def p75(g): return np.percentile(g, 75)
def p90(g): return np.percentile(g, 90)
def pos(x): return (x > 0).mean()
def ES(g): return g[g <= np.percentile(g, 5)].mean()
def MS(g): return g[g <= np.percentile(g, 5)].median()
# if the symbol has no marketcap, there is no historical data for that year
np.set_printoptions(edgeitems=3000, linewidth=1000)
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 100000
# AAPL cash flow statement only goes back to 1989 while income statement
# goes back to 1985. so columns like commonstockissued will be nan and 
# we dont want to drop the whole row so first fill nan

pd.set_option("display.precision", 2)

styles = [dict(selector="caption",
       props=[
              ("font-family", "Times New Roman"),
              ("text-align", "center"),
              ("font-size", "150%"),
              ("color", 'black')
             ]),
  dict(selector="td",
     props=[("font-family", "Times New Roman")]),
  dict(selector="th",
     props=[("font-family", "Times New Roman")])
]

def prepare_universe():
  df = pd.read_feather('master_screener.feather')
          
  df.commonStockIssued.fillna(value=0, inplace=True)
  df.operatingCashFlow.fillna(value=0, inplace=True)
  df.totalAssets.fillna(value=0, inplace=True)

  df.roa.fillna(value=0, inplace=True)
  df.d_roa.fillna(value=0, inplace=True)
  df.d_lever.fillna(value=0, inplace=True)
  df.d_current.fillna(value=0, inplace=True)
  df.d_margin.fillna(value=0, inplace=True)
  df.d_asset.fillna(value=0, inplace=True)
  # drop years where no marketcap exists
  df = df.dropna()

  # rebalance yearly
  df['bm_quintile'] = df.groupby('date').bm.transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
  df['mve_quintile'] = df.groupby('date').mve.transform(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop'))
  # one 1983 row has mve of na, drop it
  df = df[~df.mve_quintile.isna()]
  df['mve_quintile'] = df.mve_quintile.astype(int).map({0:'0:low', 1:'1:med',2:'2:high'})
  
  df['price_quintile'] = df.groupby('date').price.transform(lambda x: pd.qcut(x, 3, labels=['0:small', '1:med', '2:large'], duplicates='drop'))
  
  pgroup = {
      0:"0-1:lo",
      1:"0-1:lo",
      2:"2-7:med",
      3:"2-7:med", 
      4:"2-7:med", 
      5:"2-7:med", 
      6:"2-7:med",
      7:"2-7:med",
      8:"8-9:hi",
      9:"8-9:hi"
  }

  
  df['pscore_group'] = df.pscore.astype(int).map(pgroup)
  #df = df[df.bm_quintile==4]

  df = df.rename(columns={
    'current':'liquid',
    'd_current': 'd_liquid',
    'id_current':'id_liquid',
    'd_asset': 'd_turn',
    'id_asset':'id_turn',
    'i_shares': 'eq_offer'})
    
  df['accrual'] = df.cf - df.roa
  
  return df



@lru_cache(maxsize=10000)
def get_hist(ticker):
  feather_hist = rel(f'hist_feather/{ticker}')
  return pd.read_feather(feather_hist)

# annualized to period periods
@lru_cache(maxsize=1000000)
def computeIndividualReturn(ticker, start, end, periods=None):
    
  df = get_hist(ticker)
  firstdate = pd.to_datetime(start)
  firstiloc = max(df.index.get_indexer([firstdate], method='ffill')[0], 0)
  lastdate = pd.to_datetime(end)
  lastiloc = max(df.index.get_indexer([lastdate], method='ffill')[0], 0)
  if firstiloc == lastiloc:
    return None
  first = df.iloc[firstiloc].adjClose.item()
  if periods is None:
    periods = lastiloc - firstiloc
  # annualize the returns with compounding annualization. clip the annualization
  # so that small periods (21 days) does not blow up the returns (assume flat future)
  # e.g. fast loss: 0.1, period =20, 0.1*1=0.1
  returns = ((df.iloc[firstiloc:lastiloc].adjClose / first)**min(252.0/periods, 1) - 1.0).values
  #assert np.isnan(returns).sum() == 0  
  if returns[-1] > 300:
    print(ticker, periods, returns[-1], df.iloc[firstiloc:lastiloc].adjClose, returns)
    asdf
  return returns

# annualized
def computeReturns(portfolio, start, end):
  market_return = computeIndividualReturn(MARKET_TICKER, start, end)
  periods = len(market_return)
  
  if isinstance(portfolio, pd.Series):
    # for apply, work with series instead of dataframe
    returns = [computeIndividualReturn(portfolio.symbol, start, end, periods=periods)]
  else:
    symbols = portfolio.symbol.tolist()
    returns = [computeIndividualReturn(ticker, start, end) for ticker in symbols]
  returns = [r for r in returns if r is not None]
  
  
  if len(returns) == 0:
    return None
  
  returns2 = np.zeros((len(returns), periods))
  for i, r in enumerate(returns):
    returns2[i, periods - len(r):] = r # align to end
    returns2[i, :periods - len(r)] = r[0] # backfill front
  # equal weighted portolio returns
  eq_wght = returns2.mean(axis=0)
  
  if len(eq_wght) != len(market_return):
    print(eq_wght)
    print(market_return)
  
  ma_ret = eq_wght - market_return
  # [ABS RETURN, P0, P10, P25, P50, P75, P90, P100, AVG RET, AVG POS RET, MARKET ADJUSTED RET]
  
  
  return_vector = [ 
    eq_wght[-1], 
    (1.0*((eq_wght[-1]) > 0.0)).mean(),
    eq_wght.mean(),
    (1.0*((eq_wght.mean()) > 0.0)),
    *np.percentile(eq_wght, [0, 10, 25, 50, 75, 90, 100]).tolist(), 
    ma_ret[-1], 
    (1.0*((ma_ret[-1]) > 0.0)).mean(),
    ma_ret.mean(),
    (1.0*((ma_ret.mean()) > 0.0)),
    market_return[-1]
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
  final = pd.DataFrame(columns=['start', 'end', 'group', *return_vector_field_names, '|P|'], data=rows)
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
    hilo = hi.raw_ret.values[:_min] - lo.raw_ret.values[:_min]
    ps_hilo = ps_hi.raw_ret.values[:_min] - ps_lo.raw_ret.values[:_min]
    hiall = hi.raw_ret.values[:_min] - all.raw_ret.values[:_min]
    hilos.append(hilo[:20])
    ps_hilos.append(ps_hilo[:20])
    hialls.append(hiall[:20])
    t_hilo = stats.ttest_ind(hilo, ps_hilo)
    t_hiall = stats.ttest_ind(hi.raw_ret.values[:_min] - all.raw_ret.values[:_min], ps_hi.raw_ret.values[:_min] - ps_lo.raw_ret.values[:_min])
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
# fair, non-lookahead version.
def classify_bulls_and_bears_nonlookahead():
    df = get_hist(MARKET_TICKER)[['adjClose']]
    bulls = []
    bears = []
    state = 'bull'
    maximum = None
    minimum = None
    start = None
    starti = None
    min_transition_window = 20
    for i, row in df.reset_index().iterrows():
        current = row.adjClose
    
        if state == 'bull':
            if maximum is None:
                maximum = current
            if start is None:
                start = row.date
                starti = i
            elif (current < maximum * 0.8) and (i - starti > min_transition_window):
                bulls.append([start, row.date])
                state = 'bear'
                start = row.date
                starti = i
                minimum = current
                maximum = None
            elif maximum < current:
                maximum = current
        elif state == 'bear':
            if (current > minimum * 1.2) and (i - starti > min_transition_window):
                bears.append([start, row.date])
                state = 'bull'
                start = None
                starti = None
                minimum = None
            elif minimum > current:
                minimum = current
    if start is not None:
      if state == 'bull':
        bulls.append([start, row.date])
      else:
        bears.append([start, row.date])
    return (pd.DataFrame(data = bulls, columns=['min', 'max']).rename_axis('bearn'),
            pd.DataFrame(data = bears, columns=['min', 'max']).rename_axis('bearn'))

def classify_bulls_and_bears():
  df = get_hist(MARKET_TICKER)[['adjClose']]
  df['dd'] = df.adjClose.div(df.adjClose.cummax()).sub(1)
  df['ddn'] = ((df['dd'] < 0.) & (df['dd'].shift() == 0.)).cumsum()
  df['ddmax'] = df.groupby('ddn')['dd'].transform('min')
  df['bear'] = (df['ddmax'] < -0.2) & (df['ddmax'] < df.groupby('ddn')['dd'].transform('cummin'))
  df['bearn'] = ((df['bear'] == True) & (df['bear'].shift() == False)).cumsum()

  bears = df.reset_index().query('bear == True').groupby('bearn').date.agg(['min', 'max'])
  bulls = df.reset_index().query('bear == False').groupby('bearn').date.agg(['min', 'max'])
  return bulls, bears
  
def filter_pscores(df):
  return df[df.group.isin([f'p{i}' for i in range(10)])]
def filter_pscore_groups(df):
  return df[df.group.isin(['phi', 'plo', 'pmid'])]

def compute_bull_bear_portfolios(master, bulls, bears):
  # how do the following strategies compare
  # market - market
  # high pscore - low pscore
  # (high pscore high bm firms during bears market during bulls) - market

  filters = {
    **{f'p{i}': f'pscore == {i*1.0}' for i in range(10)},
    'plo': 'pscore <= 1.0',
    'pmid': '(pscore > 1.0) and (pscore < 8.0)',
    'phi': 'pscore >= 8.0',
    'all': '~index.isnull()',
    'market': '',
    #'adaptive': 'pscore >= 8.0',
  }
  
  max_pfolio_size=10
  trials = 1000
  market_types = {'bear':bears, 'bull':bulls}
  results = {}
  rows = []
  for market_type_name, market_type in market_types.items():
    for _, row in market_type.iterrows():
      for filtername, filter in filters.items():
        start = row["min"].strftime('%Y-%m-%d')
        end = row["max"].strftime('%Y-%m-%d')
        if filtername == 'market':# or (filtername == 'adaptive' and market_type_name == 'bear'):
          results = [computeReturns(pd.Series(data={'symbol':MARKET_TICKER}, index=['symbol']), start, end) for _ in range(trials)]
          screened_universe_size = 1
        else:#if filtername in ('plo', 'phi', 'all') or (filtername == 'adaptive' and market_type_name == 'bull'):
          df = []
          cur_year = row["min"].year - 1
          while len(df) == 0 and cur_year < row["max"].year:
            df = master[master.date==cur_year].query(filter)
            cur_year = cur_year + 1
          print(filtername, row["min"].year, 'actual earnings report year used', cur_year - 1)
          #print('p3', filtername, typenames[i], start, end)
          pfolio_size = min(len(df), max_pfolio_size)
          if not pfolio_size:
            print(row["min"].year, len(df), 'cannot run experiment')
            continue
          print(start, end, 'filter=', filtername, 'pfolio_size', pfolio_size, 'pool size', len(df))
          
          results = [computeReturns(df.sample(pfolio_size), start, end) for _ in range(trials)]
          screened_universe_size = len(df)
        # else:
          # raise Exception('invalid branch logic')
         
        results = [r for r in results if r is not None] 
        rows.extend([[start, end, market_type_name, filtername, *r, screened_universe_size ] for r in results])
  final = pd.DataFrame(columns=['start', 'end', 'market', 'group', *return_vector_field_names, '|P|'], data=rows)
  return final
def augment_individual_stock_bull_bear(master, bulls, bears):
  market_types = {'bear':bears, 'bull':bulls}
  results = {}
  rows = []
  for market_type_name, market_type in market_types.items():
    for _, row in market_type.iterrows():
        start = row["min"].strftime('%Y-%m-%d')
        end = row["max"].strftime('%Y-%m-%d')
        earnings_year = row["min"].year - 1
        basket = master[master.date==earnings_year].copy()
        basket['start'] = start
        basket['end'] = end
        basket['market'] = market_type_name
        if len(basket):
            basket[return_vector_field_names] = (basket.apply(lambda x: computeReturns(x, start, end), axis=1, result_type='expand'))
            rows.extend(basket.values.tolist())

  indiv = pd.DataFrame(columns=[*master.columns, 'start', 'end', 'market', *return_vector_field_names], data=rows).dropna()
  return indiv
  
def no_label_market_plot(bears, df, z=0):  
  df.adjClose.plot(zorder=z)
  for i, row in bears.iterrows():
    plt.fill_between(row, df.adjClose.max(), alpha=0.25, color='r', zorder=z)
def market_plot(bears, z=0):
  df = get_hist(MARKET_TICKER)[['adjClose']]
  no_label_market_plot(bears, df, z=z)
  plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
  plt.ylabel('S&P 500 Index')
  plt.title('S&P 500 Index with Bear Markets (> 20% Declines)')
  

def identify_outliers(df, m = 4.):
    df['grp_start_return_median'] = df.groupby(['start', 'market', 'return_type', 'group'])[['value']].transform('median')
    df['grp_start_return_median_dev'] = np.abs(df.value - df.grp_start_return_median)
    df['grp_start_return_median_mdev'] = df.groupby(['start', 'market', 'return_type', 'group'])[['grp_start_return_median_dev']].transform('median')
    df['outlier'] = df.grp_start_return_median_dev/df.grp_start_return_median_mdev > m
    
    # df['grp_start_return_median'] = df.groupby(['start', 'market', 'return_type', 'group'])[['value']].transform('median')
    # df['grp_start_return_median_dev'] = np.abs(df.value - df.grp_start_return_median)
    # df['grp_start_return_median_mdev'] = df.groupby(['start', 'market', 'return_type', 'group'])[['grp_start_return_median_dev']].transform('median')
    # df['outlier'] = df.grp_start_return_median_dev/df.grp_start_return_median_mdev > m
    
    return df
if __name__ == '__main__':
  #duplicate_piotroski_tests(master)
  classify_bulls_and_bears(master)