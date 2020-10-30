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
import numpy as np


def load_csv(file_path):
    df = pd.read_csv(file_path, index_col=False)
    return df


def init_base_info():
    # init info
    base_info = load_csv('../ccf_data/base_info.csv')
    print('base info shape : ', base_info.shape)

    annual_report_info = load_csv('../ccf_data/annual_report_info.csv')
    print('annual report info shape : ', annual_report_info.shape)

    tax_info = load_csv('../ccf_data/tax_info.csv')
    print('tax info shape : ', tax_info.shape)

    change_info = load_csv('../ccf_data/change_info.csv')
    print('change info shape : ', change_info.shape)

    news_info = load_csv('../ccf_data/news_info.csv')
    print('news info shape : ', news_info.shape)

    other_info = load_csv('../ccf_data/other_info.csv')
    print('other info shape : ', other_info.shape)

    train_id = load_csv('../ccf_data/entprise_info.csv')
    print('train id shape : ', train_id.shape)

    submit_id = load_csv('../ccf_data/entprise_evaluate.csv')
    print('submit dataset shape : ', submit_id.shape)

    return base_info, annual_report_info, tax_info, change_info, news_info, other_info, train_id, submit_id


def merge_fea_all():
    pass


if __name__ == '__main__':
    # init info
    base_info, annual_report_info, tax_info, change_info, news_info, other_info, train_id, submit_id = init_base_info()
