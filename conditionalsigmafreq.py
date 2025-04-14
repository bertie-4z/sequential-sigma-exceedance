## Importing packages ## 导入相关模块，顺序有所调整
from collections import Counter
import datetime as dt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from importlib import reload
import itertools
from itertools import combinations as combo_func
import json
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pandas.tseries.offsets import WeekOfMonth
import pickle
import plotly.graph_objects as go
import requests
import scipy
import statistics
import time

class ConditionalSigmaFreq:
    series: pd.Series
    std_baseline: float
    std_increment: float
    lookback_window: int
    edf: pd.DataFrame ## 'Events' dataframe, 由 布尔值 构成的 数据框
    thresholds_count = int
    thresholds: list ## list of absolute values 绝对值 构成的 串列
    possig_cpmdf: pd.DataFrame ## Positive Sigma 'Conditional Probability' Matrix dataframe 正 西格玛 的 条件概率 矩阵 dataframe; 行 = [f'P( |{b})', 列 = f'P({a}| )'
    negsig_cpmdf: pd.DataFrame ## Negative Sigma 'Conditional Probability' Matrix dataframe 负 西格玛 的 条件概率 矩阵 dataframe; 行 = [f'P( |{b})', 列 = f'P({a}| )'
    
    def __init__(self, series, std_baseline, std_increment, lookback_window, thresholds_count):
        self.series = series
        self.std_baseline = std_baseline
        self.std_increment = std_increment
        self.lookback_window = lookback_window
        self.thresholds_count = thresholds_count
        self.thresholds = [round(std_baseline + i*std_increment, 1) for i in range(thresholds_count)]
        
    def get_zscores_edf(self):
        self.edf = pd.DataFrame(self.series.rename('series'))
        self.edf[f'{self.lookback_window}mavg'] = self.series.rolling(window=self.lookback_window, closed='left').mean()
        self.edf[f'{self.lookback_window}mstd'] = self.series.rolling(window=self.lookback_window, closed='left').std()
        self.edf[f'{self.lookback_window}_zscore'] = (self.series - self.edf[f'{self.lookback_window}mavg']) / self.edf[f'{self.lookback_window}mstd']
        self.edf.drop([f'{self.lookback_window}mavg', f'{self.lookback_window}mstd'], axis=1, inplace=True)
        
        return self.edf
        
    def sigma_events_edf(self):
        threshold_cols_positive = [f'{thresh:.1f}σ' for thresh in self.thresholds]
        threshold_cols_negative = [f'-{thresh:.1f}σ' for thresh in self.thresholds]
        self.edf = self.get_zscores_edf()
        for thresh, col in zip(self.thresholds, threshold_cols_positive):
            self.edf[col] = (self.edf[f'{self.lookback_window}_zscore'] > thresh).astype(int)
        for thresh, col in zip(self.thresholds, threshold_cols_negative):
            self.edf[col] = (self.edf[f'{self.lookback_window}_zscore'] < thresh).astype(int)
        return self.edf
    
    def sigma_streaks_timestamps(self, target_std):
        self.edf = self.sigma_events_edf()
        target_std = float(target_std)
        target_sigma_col = f'{target_std:.1f}σ'
        events_col = self.edf[target_sigma_col]
        streaks = []
        in_streak = False
        start_idx = None
        prev_idx = None
        
        for idx, val in events_col.items():
            if val == 1:
                if not in_streak:
                    start_idx = idx
                    in_streak = True
            else:
                if in_streak:
                    streaks.append((start_idx, prev_idx))
                    in_streak = False
            prev_idx = idx
        # If still in a streak at the end
        if in_streak:
            streaks.append((start_idx, events_col.index[-1]))

        return streaks
        
    def compute_std_cond_prob(self, next_sigma, base_sigma, next_sigma_latency=None): ## a new sigma would only be considered to have been reached if it is within the next_sigma_latency parameter
                                                                                     ## this parameter is currently a non-relative value, but can later be adjusted to a value relative to lookback_window
        if base_sigma >= next_sigma:
            return 1
        base_streaks = self.sigma_streaks_timestamps(base_sigma) ## # list of tuples of timestamps (start, end)
        next_streaks = self.sigma_streaks_timestamps(next_sigma)        
        next_sigma_exceeded_counter = 0    
        for bs_start, bs_end in base_streaks:
            for ns_start, ns_end in next_streaks:
                if bs_start <= ns_start and ns_end <= bs_end:
                    if not next_sigma_latency:
                        next_sigma_exceeded_counter += 1
                        break
                    elif next_sigma_latency:
                        if (ns_start - bs_start) <= timedelta(minutes = next_sigma_latency):
                            next_sigma_exceeded_counter += 1
                            break

        if len(base_streaks) == 0:
            return np.nan

        return next_sigma_exceeded_counter/len(base_streaks)
        
    
    def cond_prob_df(self): ## returns a cpdf
        positive_cols = self.thresholds
        negative_cols = [-x for x in self.thresholds]
        
        self.possig_cpmdf = pd.DataFrame(index=[f'P( | {b}σ)' for b in positive_cols],
                           columns=[f'P({a}σ | )' for a in positive_cols])
        self.negsig_cpmdf = pd.DataFrame(index=[f'P( | {b}σ)' for b in negative_cols],
                           columns=[f'P({a}σ | )' for a in negative_cols])
        
        for a in positive_cols:
            for b in positive_cols:                
                cp = self.compute_std_cond_prob(a,b)
                self.possig_cpmdf.at[f'P( | {b}σ)', f'P({a}σ | )'] = cp
        
        for a in negative_cols:
            for b in negative_cols:                
                cp = self.compute_std_cond_prob(a,b)
                self.negsig_cpmdf.at[f'P( | {b}σ)', f'P({a}σ | )'] = cp
        
        return self.possig_cpmdf, self.negsig_cpmdf
        
        
    def draw_histograms(self, base_sigma): ## 绘制 所有 超过 基准σ 的 σ事件 的 直方图，并 按 基准σ 进行 缩放
        self.possig_cpmdf, self.negsig_cpmdf = self.cond_prob_df()
        base_sigma_label = f'P( | {base_sigma:.1f}σ)' ## 基准σ 代表的 是 每行的 索引值 ## base sigma is row's index value 

        if base_sigma > 0:
            if base_sigma not in self.thresholds:
                raise ValueError(f"{base_sigma} not found in DataFrame index.")
            row = self.possig_cpmdf.loc[base_sigma_label]
            filtered_data = {
                col: val for col, val in row.items()
                if float(col.replace('P(', '').replace('σ | )', '')) >= base_sigma
                        }
            graph_data = pd.Series(filtered_data) ## 过滤后的  now the filtered column names are the index values
            x_labels = [col.replace("P(", "").replace("σ | )", "σ") for col in graph_data.index]
            plt.figure(figsize=(10, 6))
            plt.bar(x_labels, graph_data.values, color='steelblue', edgecolor='black')
            # Add value labels
            bars = plt.bar(x_labels, graph_data.values, color='steelblue', edgecolor='black')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                         ha='center', va='bottom', fontsize=10)
            plt.title(f"Conditional Probabilities Given {base_sigma_label}")
            plt.xlabel("Next Sigma Events")
            plt.ylabel("Conditional Probability")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show();

            
        if base_sigma < 0:
            if base_sigma not in [-x for x in self.thresholds]:
                raise ValueError(f"{base_sigma} not found in DataFrame index.")
                        
            row = self.negsig_cpmdf.loc[base_sigma_label]
            filtered_data = {
                col: val for col, val in row.items()
                if float(col.replace('P(', '').replace('σ | )', '')) <= base_sigma
                        }
            graph_data = pd.Series(filtered_data) ## 过滤后的  now the filtered column names are the index values
            x_labels = [col.replace("P(", "").replace("σ | )", "σ") for col in graph_data.index]
            plt.figure(figsize=(10, 6))
            plt.bar(x_labels, graph_data.values, color='steelblue', edgecolor='black')
            # Add value labels
            bars = plt.bar(x_labels, graph_data.values, color='steelblue', edgecolor='black')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                         ha='center', va='bottom', fontsize=10)
            plt.title(f"Conditional Probabilities Given {base_sigma_label}")
            plt.xlabel("Next Sigma Events")
            plt.ylabel("Conditional Probability")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show();

            
    def draw_curves(self, cpmdf):
        plt.figure(figsize=(12, 6))
        for idx, (row_label, row) in enumerate(cpmdf.iterrows()):
            base_sigma = float(row_label.replace("P( | ", "").replace("σ)", ""))
            # Filter relevant columns: those whose sigma > base_sigma
            filtered_data = {
                col: val for col, val in row.items()
                if float(col.replace("P(", "").replace("σ | )", "")) > base_sigma
            }
            if not filtered_data:
                continue
            graph_data = pd.Series(filtered_data)
            x_values = [
                float(col.replace("P(", "").replace("σ | )", "")) - base_sigma
                for col in graph_data.index
                        ]
            plt.plot(x_values, graph_data.values, marker='o', label=f'{row_label} 阈值')
            
        plt.title("Conditional Probability Curves: P(next σ | base σ)")
        plt.xlabel("Δ Sigma (next σ - base σ)")
        plt.ylabel("Conditional Probability")
        plt.xticks(range(1, 5))  # Adjust based on your max delta
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Condition (Base σ)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show();
                        
