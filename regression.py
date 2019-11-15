import pandas as pd
import numpy as np

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
    arolla = pd.read_excel("dataset/original_excel/pluie_Arolla.xlsx")
    arollaTemp = pd.read_excel("dataset/original_excel/temperature_Arolla.xlsx")
    tsijiore = pd.read_csv("dataset/debit_Tsijiore.csv")
    bertol = pd.read_csv("dataset/Bertol_inferieur.csv")
    tsijiore_stack = stack_data(tsijiore)

    arolla_stack = stack_data(arolla)
    diff = arolla_stack.diff()

    change_of_year = [False] + list((diff.index[1:] - diff.index[0:-1]) > pd.Timedelta(days=1))

    diff[arolla_stack == 0] = np.NaN
    diff[change_of_year] = np.NaN
    diff[diff < -40] = diff[diff < -40] + 50
    diff[np.abs(diff) < 0.08] = 0
    diff[diff < 0.0] = 0
    diff[diff > 15] = np.NaN

    change_of_year = [False] + list((diff.index[1:] - diff.index[0:-1]) > pd.Timedelta(days=1))
    arollaTemp_stack = stack_data(arollaTemp)
    arollaTemp_stack[change_of_year] = np.NaN

