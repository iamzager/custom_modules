"""
version = 0.1
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def annotate_bars(plot, ha='center', va='bottom', precision=None):
    for p in plot.patches:
        x = p.get_bbox().get_points()[:,0]
        y = p.get_bbox().get_points()[1,1]
        caption = str(round(p.get_height(), precision))
        plot.annotate(caption, (x.mean(), y), ha=ha, va=va)