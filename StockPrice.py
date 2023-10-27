def PriceDownload(ticker, starttime):
 # This function takes in a ticker of some stock, and some time indacting
 # how many trading days in the past you want the data to go back.
 # It will return a numpy array with just the price at open
  try:
      Model_Data = yf.download(ticker, start=date.today()-timedelta(starttime), end=date.today(), progress=False) #Collect all the data
  except KeyError: #If the ticker was typed wrong or does not exist
        return False
  if len(Model_Data)==0: #sometimes it returns no data
        return False
    
  Model_Data = Model_Data["Open"]#Strip down the data set to just the price at open
  Model_Data.dropna()
  Model_Data=Model_Data.to_numpy() #Dataframe to list of historic, or taining data.
  return Model_Data
PriceDownload('AAPL',365)
