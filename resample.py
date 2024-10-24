# !pip install pandas
import pandas as pd
import matplotlib.pyplot as plt

# read all csv files
raw_df_41008 = pd.read_csv('Pra Data/Pra Data/41008.csv')
raw_df_41009 = pd.read_csv('Pra Data/Pra Data/41009.csv')
raw_df_41010 = pd.read_csv('Pra Data/Pra Data/41010.csv')
raw_df_42022 = pd.read_csv('Pra Data/Pra Data/42022.csv')
raw_df_42036 = pd.read_csv('Pra Data/Pra Data/42036.csv')
raw_df_fwyf1 = pd.read_csv('Pra Data/Pra Data/fwyf1.csv')
raw_df_smkf1 = pd.read_csv('Pra Data/Pra Data/smkf1.csv')
raw_df_venf1 = pd.read_csv('Pra Data/Pra Data/venf1.csv')


# combine YY, MM, DD and hh to "Date/Time"
raw_df_41008['Date/Time'] = pd.to_datetime(raw_df_41008[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_41009['Date/Time'] = pd.to_datetime(raw_df_41009[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_41010['Date/Time'] = pd.to_datetime(raw_df_41010[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_42022['Date/Time'] = pd.to_datetime(raw_df_42022[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_42036['Date/Time'] = pd.to_datetime(raw_df_42036[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_fwyf1['Date/Time'] = pd.to_datetime(raw_df_fwyf1[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_smkf1['Date/Time'] = pd.to_datetime(raw_df_smkf1[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))
raw_df_venf1['Date/Time'] = pd.to_datetime(raw_df_venf1[['YY', 'MM', 'DD', 'hh']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour'}))

# calculate hourly mean value and delete YY, MM, DD, hh, mm
raw_df_41010 = raw_df_41010.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_41010= raw_df_41010.set_index('Date/Time').resample('2H').mean().reset_index()
# print(df_41010)
raw_df_41008 = raw_df_41008.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_41008= raw_df_41008.set_index('Date/Time').resample('2H').mean().reset_index()

raw_df_41009 = raw_df_41009.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_41009= raw_df_41009.set_index('Date/Time').resample('2H').mean().reset_index()

raw_df_42022 = raw_df_42022.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_42022= raw_df_42022.set_index('Date/Time').resample('2H').mean().reset_index()

raw_df_42036 = raw_df_42036.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_42036= raw_df_42036.set_index('Date/Time').resample('2H').mean().reset_index()

raw_df_fwyf1 = raw_df_fwyf1.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_fwyf1= raw_df_fwyf1.set_index('Date/Time').resample('2H').mean().reset_index()

raw_df_smkf1 = raw_df_smkf1.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_smkf1= raw_df_smkf1.set_index('Date/Time').resample('2H').mean().reset_index()

raw_df_venf1 = raw_df_venf1.drop(columns=['YY', 'MM', 'DD', 'hh','mm'])
df_venf1= raw_df_venf1.set_index('Date/Time').resample('2H').mean().reset_index()

df_41008.to_csv('Hourly_Data/41008.csv', index=False)
df_41009.to_csv('Hourly_Data/41009.csv', index=False)
df_41010.to_csv('Hourly_Data/41010.csv', index=False)
df_42022.to_csv('Hourly_Data/42022.csv', index=False)
df_42036.to_csv('Hourly_Data/42036.csv', index=False)
df_fwyf1.to_csv('Hourly_Data/fwyf1.csv', index=False)
df_smkf1.to_csv('Hourly_Data/smkf1.csv', index=False)
df_venf1.to_csv('Hourly_Data/venf1.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(df_41010['Date/Time'], df_41010['WVHT'])
plt.title('Wave Height Over Time (Station 41010)')
plt.xlabel('Time')
plt.ylabel('WVHT')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_41010['Date/Time'], df_41010['MWD'])
plt.title('Mean Wave Direction Over Time (Station 41010)')
plt.xlabel('Time')
plt.ylabel('MWD')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_41010['Date/Time'], df_41010['ATMP'])
plt.title('Air Temperature Over Time (Station 41010)')
plt.xlabel('Time')
plt.ylabel('ATMP')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_41010['Date/Time'], df_41010['WTMP'])
plt.title('Water Temperature Over Time (Station 41010)')
plt.xlabel('Time')
plt.ylabel('WTMP')
plt.show()