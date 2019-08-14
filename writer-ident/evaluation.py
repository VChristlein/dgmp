import os
import torch
import numpy as np
import pandas as pd


def avg_precision(ranking, target, relevant_docs):
    hits = 0
    count = 0
    precision = 0
    # print(ranking.head())
    for e in ranking:
        count += 1
        if int(e) == target:
            hits = hits + 1
            precision += hits / count
            if hits == relevant_docs:
                break
    return precision / relevant_docs


def mean_avg_precision(rankings: pd.DataFrame, relevant_docs: int, starting_col=2):
    c = 0
    avg_precs = 0
    for i in range(starting_col, len(rankings.columns), 2):
        ranking = rankings[rankings.columns[i]]
        target = int(ranking[0])
        ranking = ranking[1:]
        if i % 500 == 0:
            sim = rankings[rankings.columns[i - 1]]

        c += 1
        avg_precs += avg_precision(ranking, target, relevant_docs)

    return avg_precs / c


def mean_average_precision_of_rankings(ranking_dir):
    mAPs = []
    ranking_names = []
    ranking_files = (f for f in os.listdir(ranking_dir)
                     if os.path.isfile(os.path.join(ranking_dir, f)))
    for ranking_file in ranking_files:
        ranking_names.append(ranking_file)
        df = pd.read_csv(os.path.join(ranking_dir, ranking_file))
        mAPs.append(mean_avg_precision(df, 2))
    result_df = pd.DataFrame()
    result_df['mAP'] = mAPs
    result_df['params'] = ranking_names
    return result_df
