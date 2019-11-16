import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def stack_data(arolla):
    arolla = arolla.iloc[1:,:]
    arolla2019 = arolla.iloc[:,0]
    arolla2018 = arolla.iloc[:,1]
    arolla2018.index = arolla2018.index + pd.DateOffset(years=-1)
    arolla2017 = arolla.iloc[:,2]
    arolla2017.index = arolla2017.index + pd.DateOffset(years=-2)

    arolla2016 = arolla.iloc[:,3]
    arolla2016.index = arolla2016.index + pd.DateOffset(years=-3)

    arolla2015 = arolla.iloc[:,4]
    arolla2015.index = arolla2015.index + pd.DateOffset(years=-4)
    return pd.concat([arolla2015,arolla2016,arolla2017,arolla2018,arolla2019])


if __name__ == '__main__':
    arolla = pd.read_csv("dataset/clean_data/debitTsijiore.csv", names=["time", "diff"], index_col=0)
    arolla.plot()
    plt.show()

