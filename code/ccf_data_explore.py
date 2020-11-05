#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ccf_data_explore.py
@Contact :   caoshuaiyi@brdc.icbc.com.cn, shuaiyicao@foxmail.com
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/10/30 17:07   shuaiyicao    1.0         None
"""
import pandas as pd
from functools import reduce
import numpy as np


def load_csv(file_path):
    df = pd.read_csv(file_path, index_col=False, encoding='utf-8')
    return df


def init_base_info():
    # init info/
    # base_info = load_csv('../ccf_data/base_info.csv')
    base_info = load_csv('../code/fea_explore/base_info.csv')
    print('base info shape : ', base_info.shape)

    annual_report_info = load_csv('../ccf_data/annual_report_info.csv')
    print('annual report info shape : ', annual_report_info.shape)

    tax_info = load_csv('../ccf_data/tax_info.csv')
    print('tax info shape : ', tax_info.shape)

    change_info = load_csv('../ccf_data/change_info.csv')
    print('change info shape : ', change_info.shape)

    news_info = load_csv('../code/fea_explore/news_info.csv')
    print('news info shape : ', news_info.shape)

    other_info = load_csv('../code/fea_explore/other_info.csv')
    print('other info shape : ', other_info.shape)

    train_id = load_csv('../ccf_data/entprise_info.csv')
    print('train id shape : ', train_id.shape)

    submit_id = load_csv('../ccf_data/entprise_evaluate.csv')
    print('submit dataset shape : ', submit_id.shape)

    return base_info, annual_report_info, tax_info, change_info, news_info, other_info, train_id, submit_id


def merge_fea_all(train_id, base_info, other_info, news_info):
    # merge base info
    # merge_fea = pd.merge(base_info, train_id, on=['id'], how='left')
    # merge_fea = pd.merge(train_id, base_info, on=['id'], how='left')
    # merge_fea = pd.merge(merge_fea, other_info, on=['id'], how='left')

    dfs = [train_id, base_info, other_info, news_info]
    merge_fea = reduce(lambda left, right: pd.merge(left, right, on=['id'], how='left'), dfs)

    convert_col = ['legal_judgment_num', 'brand_num', 'patent_num']
    for col in convert_col:
        merge_fea[col].fillna(-1, inplace=True)
        other_info[col] = other_info[col].astype('float64')

    print('Merge Fea shape : ', merge_fea.shape)
    print(merge_fea.columns)
    return merge_fea


def drop_na_cols(df, expect):
    # df shape
    rows, cols = df.shape[0], df.shape[1]

    # remove expect
    subset = list(df.columns)
    for col in expect:
        subset.remove(col)

    # df drop na cols
    df_isna = df.isna().sum()
    thresh = 0.8
    drop_list = []

    for col in subset:
        if df_isna[col] > rows * thresh:
            drop_list.append(col)

    for col in drop_list:
        del df[col]

    return df, drop_list


def drop_single_cols(df, expect):
    # df shape
    rows, cols = df.shape[0], df.shape[1]

    # remove expect
    subset = list(df.columns)
    for col in expect:
        subset.remove(col)

    cnt = 0
    drop_list = []
    thresh = 0.8

    for col in subset:
        cnt = len(df[col].value_counts())
        if cnt >= thresh * rows:
            del df[col]
            drop_list.append(col)

    return df, drop_list


if __name__ == '__main__':
    # init info
    base_info, annual_report_info, tax_info, change_info, news_info, other_info, train_id, submit_id = init_base_info()

    # get merge fea
    merge_fea = merge_fea_all(train_id, base_info, other_info, news_info)

    # remove na and null
    merge_fea, drop_list_na = drop_na_cols(merge_fea, expect=['id', 'label', 'legal_judgment_num', 'brand_num',
                                                              'patent_num', 'news_sum', 'news_pos_sum', 'news_mid_sum',
                                                              'news_neg_sum'])

    # remove single col
    merge_fea, drop_list_single = drop_single_cols(merge_fea, expect=['id', 'label', 'legal_judgment_num', 'brand_num',
                                                              'patent_num', 'news_sum', 'news_pos_sum', 'news_mid_sum',
                                                              'news_neg_sum'])

    # split year and month
    # merge_fea['year'] = merge_fea['opfrom'].apply(lambda x: int(x.split('-')[0]))
    # merge_fea['month'] = merge_fea['opfrom'].apply(lambda x: int(x.split('-')[1]))

    # del cols
    del_list = ['opfrom', 'opto', 'oploc']
    for col in del_list:
        del merge_fea[col]

    print(merge_fea.columns)

    merge_fea.fillna(0, inplace=True)

    merge_fea.to_csv('./fea/fea_1105.csv', index=False)
