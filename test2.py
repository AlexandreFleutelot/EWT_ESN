from market_profile import MarketProfile
import pandas_datareader as data
amzn = data.get_data_yahoo('AMZN', '2019-12-01', '2019-12-31')

mp = MarketProfile(amzn)
mp_slice = mp[amzn.index.min():amzn.index.max()]
print(mp_slice.profile)