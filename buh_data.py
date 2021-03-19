import numpy as np
#from my_scripts import annotate_bars
import seaborn as sns
import matplotlib.pyplot as plt


def plot_chars(data, chars, normalize=True, figsize=(20, 6), precision=3):
    result = np.array([])
    for col in data.drop('year', axis=1):
        result = np.r_[result, contains_chars(data[col].dropna(), chars, normalize=normalize)]
    result = result.reshape(data.drop('year', axis=1).columns.shape[0], len(chars))
    for idx, char in enumerate(chars):
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=data.drop('year', axis=1).columns,\
                                                 y=result.T[idx])
    plt.grid(True)
    plt.title(f'Number of occurences of "{char}"')
    plt.xticks(rotation=0)
    #annotate_bars(ax, precision=precision)

def contains_chars(data, chars, normalize=False):
    result = []
    length = data.shape[0]
    for char in chars:
        try:
            if normalize:
                result.append(round(data.str.contains(char)\
                              .value_counts()[True] / length, 3))
            else: 
                result.append(data.str.contains(char)\
                              .value_counts()[True])
        except KeyError:
            result.append(0)        
    return result
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
