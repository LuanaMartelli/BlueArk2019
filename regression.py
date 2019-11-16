import pandas as pd
import numpy as np
import statsmodels.api as sm
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

def temperature(tem):
    tem = stack_data(tem)
    change_of_year = [False] + list((tem.index[1:] - tem.index[0:-1]) > pd.Timedelta(days=1))
    tem[change_of_year] = np.NaN
    return tem


def pluie(arollaPluie):
    arolla_stack = stack_data(arollaPluie)
    diff = arolla_stack.diff()

    change_of_year = [False] + list((diff.index[1:] - diff.index[0:-1]) > pd.Timedelta(days=1))

    diff[arolla_stack == 0] = np.NaN
    diff[change_of_year] = np.NaN
    diff[diff < -40] = diff[diff < -40] + 50
    diff[np.abs(diff) < 0.08] = 0
    diff[diff < 0.0] = 0
    diff[diff > 15] = np.NaN
    diff[change_of_year] = np.NaN
    return diff

def clean_debit(debit):
    debit = stack_data(debit)
    debit = debit.replace(0,np.NaN).ffill()
    for i in range(100):
        debit[debit.rolling(window=10).mean() - debit > 0.08] = debit.shift(1)[debit.rolling(window=10).mean() - debit > 0.08]
    change_of_year = [False] + list((debit.index[1:] - debit.index[0:-1]) > pd.Timedelta(days=1))
    debit[change_of_year] = np.NaN
    return debit

def ensoleillement(soleil):
    shine = stack_data(soleil)
    shine[shine < 1] = 0
    return shine

if __name__ == '__main__':
    bertolDebit = pd.read_excel("dataset/original_excel/debit_Bertol_inferieur.xlsx")
    bertolDebitClean = clean_debit(bertolDebit)
    bertolDebitClean.to_csv("dataset/clean_data/debit_bertol.csv")

    arollaPluie = pd.read_excel("dataset/original_excel/pluie_Arolla.xlsx")
    arolla_pluie_diff = pluie(arollaPluie)
    arolla_pluie_diff.to_csv("dataset/clean_data/arollaPluie_diff.csv")

    bricolaPluie = pd.read_excel("dataset/original_excel/pluie_Bricola.xlsx")
    bricola_pluie_diff = pluie(bricolaPluie)
    bricola_pluie_diff.to_csv("dataset/clean_data/pluie_bricola.csv")

    gorneraPluie = pd.read_excel("dataset/original_excel/pluie_Gornera.xlsx")
    gornera_pluie_diff = pluie(gorneraPluie)
    gornera_pluie_diff.to_csv("dataset/clean_data/pluie_gornera.csv")

    tsijiore = pd.read_excel("dataset/original_excel/debit_Tsijiore.xlsx")
    tsijiore_stack_clean = clean_debit(tsijiore)
    tsijiore_stack_clean.to_csv("dataset/clean_data/debitTsijiore.csv")

    arollaTemp = pd.read_excel("dataset/original_excel/temperature_Arolla.xlsx")
    arollaTemp_stack = temperature(arollaTemp)
    arollaTemp_stack.to_csv("dataset/clean_data/arollaTemp.csv")

    edelweissDebit = pd.read_excel("dataset/original_excel/debit_edelweiss.xlsx")
    edelweissDebit = clean_debit(edelweissDebit)
    edelweissDebit.to_csv("dataset/clean_data/edelweissDebit.csv")

    zmuttTemp = pd.read_excel("dataset/original_excel/temperature_Zmutt.xlsx")
    zmuttTemp = temperature(zmuttTemp)
    zmuttTemp.to_csv("dataset/clean_data/zmuttTemp.csv")

    pluie_Findelen = pd.read_excel("dataset/original_excel/pluie_Findelen.xlsx")
    pluie_Findelen = pluie(pluie_Findelen)
    pluie_Findelen.to_csv("dataset/clean_data/pluie_Findelen.csv")

    rayonnement_Findelen = pd.read_excel("dataset/original_excel/rayonnement_Findelen.xlsx")
    rayonnement_Findelen = ensoleillement(rayonnement_Findelen)
    rayonnement_Findelen.to_csv("dataset/clean_data/rayonnement_Findelen.csv")


    pd.concat([arollaTemp_stack, tsijiore_stack_clean], axis=1).dropna()

    Xy = pd.concat([arollaTemp_stack.shift(12).replace(0,np.NaN), np.sqrt(tsijiore_stack_clean)], axis=1).dropna().astype(float)
    res = sm.OLS(np.asarray(Xy.iloc[:, -1]), sm.add_constant(np.asarray(Xy.iloc[:, 0:-1]))).fit()
    res.summary()
    Xy.corr()

    Xy.plot.scatter(0, 1, alpha=0.01)
    plt.xlabel('temperature')
    plt.ylabel('debit')
    plt.title('Tsidjore')

    plt.plot(np.sort(Xy.iloc[:, 0]), res.params[1] * np.sort(Xy.iloc[:, 0]) + res.params[0])
    # diff = arolla_stack.diff()
    #
    # change_of_year = [False] + list((diff.index[1:] - diff.index[0:-1]) > pd.Timedelta(days=1))
    #
    # diff[arolla_stack == 0] = np.NaN
    # diff[change_of_year] = np.NaN
    # diff[diff < -40] = diff[diff < -40] + 50
    # diff[np.abs(diff) < 0.08] = 0
    # diff[diff < 0.0] = 0
    # diff[diff > 15] = np.NaN


    # change_of_year = [False] + list((diff.index[1:] - diff.index[0:-1]) > pd.Timedelta(days=1))
    # arollaTemp_stack = stack_data(arollaTemp)
    # arollaTemp_stack[change_of_year] = np.NaN



    # tsijiore_stack = stack_data(tsijiore)
    # tsijiore_stack = tsijiore_stack.replace(0,np.NaN).ffill()
    # var = 10 * tsijiore_stack.diff().rolling(window=300).std() < tsijiore_stack.diff()
    # for i in range(100):
    #     tsijiore_stack[tsijiore_stack.rolling(window=10).mean() - tsijiore_stack > 0.08] = tsijiore_stack.shift(1)[tsijiore_stack.rolling(window=10).mean() - tsijiore_stack > 0.08]


    #    tsijiore_stack[tsijiore_stack.diff() < -0.1] = tsijiore_stack.shift(1)[tsijiore_stack.diff() < -0.1]
    #    tsijiore_stack[var] = tsijiore_stack.shift(1)[var]
