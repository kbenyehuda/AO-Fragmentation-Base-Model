import pandas as pd

def read_feats_file(filename):

    # filename = 'final_exp_3.xlsx'

    xls = pd.ExcelFile(filename)
    df302 = pd.read_excel(xls, 'donor 302')
    df305 = pd.read_excel(xls, 'donor 305')
    df306 = pd.read_excel(xls, 'donor 306')
    df308 = pd.read_excel(xls, 'donor 308')
    df309 = pd.read_excel(xls, 'donor 309')
    df310 = pd.read_excel(xls, 'donor 310')
    df311 = pd.read_excel(xls, 'donor 311')
    df312 = pd.read_excel(xls, 'donor 312')
    df314 = pd.read_excel(xls, 'donor 314')

    donors = [302,305,306,308,309,310,311,312,314]
    all_dfs = [df302,df305,df306,df308,df309,df310,df311,df312,df314]

    return all_dfs,donors