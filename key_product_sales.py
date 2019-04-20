#importing all the packages
import numpy as np
import itertools
import statsmodels.api as sm
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

data = pd.read_csv("product_distribution_training_set.txt", delimiter='\t',header=None)
data=data.T
product_id=data.iloc[0:1,]
idn=np.array(product_id)

for i in range (100):
    product=data.loc[1:,i]
    train=product[0:118]
    test=product[118: ]
    product_id=data.iloc[0:1,]
    idn=np.array(product_id)
    id=idn[0]
    print("product no",i,"product id",id[i])
    # applying the SARIMA model 
    mod = sm.tsa.statespace.SARIMAX(train,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    #results.plot_diagnostics(figsize=(16, 8))
    #plt.show()
    pred = results.get_prediction(20)
    pred_ci = pred.conf_int()
    #train.plot(label='observed',color='red')
    #ax = train.plot(label='observed')
    #pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    #ax.fill_between(pred_ci.index,
     #           pred_ci.iloc[:, 0],
      #          pred_ci.iloc[:, 1], color='k', alpha=.2)
    #ax.set_xlabel('Days')
    #ax.set_ylabel('Sales')
    #plt.legend()
    #plt.show()
	
    #predict the model to 28 days further
    pred_uc = results.get_forecast(steps=29)
    pred_ci = pred_uc.conf_int()
    #ax = train.plot(label='observed', figsize=(14, 7))
    #pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    #ax.fill_between(pred_ci.index,
     #           pred_ci.iloc[:, 0],
      #          pred_ci.iloc[:, 1], color='k', alpha=.25)
    #ax.set_xlabel('Days')
    #ax.set_ylabel('sales')
    #plt.plot(test,color="green")
    #plt.plot(test)
    #plt.legend()
    #plt.show()
    print(pred_uc.predicted_mean)
    asd=round(pred_uc.predicted_mean)
    out=np.array(asd)
    f= open("predicted_result.txt","a+")
    f.write("\n")
    f.write(str(id[i]))
    f.write(" ")
    for i in range(29):
        output=out[i]
        f.write(str(abs(round(int(output))))+" ")
		

f.close()
    

