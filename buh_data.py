import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from custom_modules.plotting import annotate_bars
from custom_modules.general import trim

def plot_chars(data, chars, normalize=True, figsize=(20, 6), precision=3):
    result = np.array([])
    df = data
    if 'year' in df.columns:
        df = df.drop('year', axis=1)
    if 'inn' in df.columns:
        df = df.drop('inn', axis=1)

    for col in df:
        result = np.r_[result, contains_chars(df[col].dropna(), chars, normalize=normalize, precision=precision)]
    result = result.reshape(df.columns.shape[0], len(chars))
    for idx, char in enumerate(chars):
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=df.columns, y=result.T[idx])
        annotate_bars(ax, precision=precision)
        plt.grid(True)
        plt.title(f'Number of occurences of "{char}"')
        plt.xticks(rotation=0)
    

def contains_chars(data, chars, normalize=False, precision=3):
    result = []
    length = data.shape[0]
    for char in chars:
        try:
            if normalize:
                result.append(round(data.str.contains(char)\
                              .value_counts()[True] / length, precision))
            else: 
                result.append(int(data.str.contains(char)\
                              .value_counts()[True]))
        except KeyError:
            result.append(0)        
    return result

def clear_df_test(df, cols_to_add, cols_to_sub):
    new_df = df.copy()
    new_df[cols_to_add] = new_df[cols_to_add].applymap(\
                                lambda x :  clear_value_test(x, True), na_action='ignore')
    new_df[cols_to_sub] = new_df[cols_to_sub].applymap(\
                                lambda x :  clear_value_test(x, False), na_action='ignore')

    return new_df
def clear_value_test(value, add):
    if value.find('(') != -1:
        if add:
            output = - float(value.strip('()'))
        if not add:
            output = float(value.strip('()'))
    else:
        if add:
            output = float(value)
        if not add:
            output = - float(value)
    return output

def clear_df(df, cols_w_neg_values, cols_wo_neg_values):
    new_df = df.copy()
    new_df[cols_w_neg_values] = new_df[cols_w_neg_values].applymap(\
                                lambda x :  clear_value(x, True), na_action='ignore')
    new_df[cols_wo_neg_values] = new_df[cols_wo_neg_values].applymap(\
                                lambda x :  clear_value(x, False), na_action='ignore')

    return new_df
def clear_value(value, negative):
    if value.find('(') != -1:
        if negative:
            output = - float(value.strip('()'))
        if not negative:
            output = float(value.strip('()'))
    else:
        output = float(value)
    return output
def zero_to_nan(value):
    if value == 0 :
        return None
    else: 
        return value
def balance_fill(balance, fill_zeros=False):
    """
    Converts all zeros in balance to NaN
    Counts totals where there are none
    If nedeed fills NaNs with zeros   
    
    fill_zeros: bool - if True fills NaNs with zeros in rows
    with at least one value (required to use balance_deviations)
    """
    balance_df = balance.applymap(zero_to_nan)
    if not fill_zeros:
        print('''Warning: If you want to use balance_deviations function, set fill_zeros to True''')
    for i in range(1, 6):
        prefix = f'1{i}'
        columns = cols[i]
        non_empty_rows = np.any(~balance_df[columns].isna(), axis=1)
        balance_df.loc[(balance_df[prefix + '00'].isna()) & (non_empty_rows), prefix + '00'] = \
                       balance_df[columns].sum(axis=1)
        if fill_zeros:
            balance_df.loc[non_empty_rows, columns] = \
            balance_df.loc[non_empty_rows, columns].fillna(0)        
    return balance_df

def balance_deviations(balance_df): 
    """
    Counts deviations in balance totals
    """    
    output = balance_df.eval('''
    Актив_Пассив = `1600` - `1700`
    Расчет_1700_итоги = `1700` - `1300` - `1400` - `1500`
    Расчет_1600_итоги = `1600` - `1100` - `1200`
    Расчет_1700_статьи = `1700` \
        - `1310` - `1320` - `1340` - `1350` - `1360` - `1370`\
        - `1410` - `1420` - `1430` - `1450`\
        - `1510` - `1520` - `1530` - `1540` - `1550`    
    Расчет_1600_статьи = `1600`\
        - `1110` - `1120` - `1130` - `1140` - `1150` - `1160` \
        - `1170` - `1180` - `1190`\
        - `1210` - `1220` - `1230` - `1240` - `1250` - `1260`
    Расчет_1100 = `1100` - `1110` - `1120` - `1130` - `1140` - `1150` - `1160` \
        - `1170` - `1180` - `1190`
    Расчет_1200 = `1200` - `1210` - `1220` - `1230` - `1240` - `1250` - `1260`
    Расчет_1300 = `1300` - `1310` - `1320` - `1340` - `1350` - `1360` - `1370`
    Расчет_1400 = `1400` - `1410` - `1420` - `1430` - `1450`
    Расчет_1500 = `1500` - `1510` - `1520` - `1530` - `1540` - `1550`
    ''').iloc[:, -10:]
    return output
def deviation_report(deviations_df, tol=0, hist=False):   
    report = pd.DataFrame(columns=deviations_df.columns)
    if hist:
        if report.shape[1] % 3 == 0:          
            nrows = report.shape[1] / 3
        else:
            nrows = (report.shape[1] / 3) + 1
        plt.figure(figsize=(15,15))
        for idx, col in enumerate(deviations_df.columns):
            plt.subplot(int(nrows), 3, idx+1)
            ax = plt.hist(deviations_df[np.abs(deviations_df[col]) > 0][col])
            plt.xlabel('Отклонение', fontsize=8)
            plt.ylabel('Количество', fontsize=8)
            plt.title(col)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

    counts = deviations_df.apply(lambda x : (np.abs(x) > tol).sum().astype('int'))
    means = deviations_df.apply(lambda x : x[np.abs(x) > tol].mean())
    medians = deviations_df.apply(lambda x : x[np.abs(x) > tol].median())
    maxs = deviations_df.max() 
    mins = deviations_df.min()
    report = pd.DataFrame(np.vstack([counts, means, medians, mins, maxs]),\
                          index=['counts > tol', 'mean > tol', 'median > tol', 'min', 'max'],\
                          columns=deviations_df.columns)
    return report
def choose_random(df, choose_from=None):
    if type(choose_from) != 'NoneType':        
        random_index = np.random.choice(choose_from)
    else:
        random_index = np.random.choice(df.index)
    return df.loc[random_index, :].T, random_index

def plot_hist(df, with_zeros=False, figsize=(12,8), log=False, outliers=0, ncols=3):
    """
    with_zeros: bool - if False, drops zeros as well as NaN values
    kog: bool - applies np.log1p to POSITIVE values
    outliers: int - number of standard deviations to use while trimming outliers
    """
    if df.shape[1] % ncols == 0:          
        nrows = df.shape[1] / ncols
    else:
        nrows = (df.shape[1] / ncols) + 1
    plt.figure(figsize=figsize)
    for idx, col in enumerate(df.columns):
        plt.subplot(int(nrows), ncols, idx+1)        
        data = df[col]
        num = round(100 * data.notna().mean(), 2)
        data = data.dropna()

        if not with_zeros:
            data = data[data != 0]
        if outliers:
            data = data[trim(data, outliers)]
        if log:
            if data.min() < 0:
                positive_rate = round(100 * (data >= 0).mean(), 3)
                neg_data = np.log1p(np.abs(data[data < 0]))                 
                print(f'Making separate hists for positive {positive_rate}%  and negative {round(100-positive_rate, 3)}% values in {col}')
                plt.hist(neg_data, zorder=2, density=False, rwidth=0.9, color='r', alpha=1)
#                 plt.legend()
                data = np.log1p((data[data >= 0]) )
            else:
                data = np.log1p(data)            
        ax = plt.hist(data, zorder=2, density=False, rwidth=0.9, color='g', alpha=0.7)
        plt.title(f'Непустых значений: {num}%')
        plt.xlabel(col)
        plt.grid(True, zorder=0)
        
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
