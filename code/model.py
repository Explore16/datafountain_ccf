#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py
@Contact :   caoshuaiyi@brdc.icbc.com.cn, shuaiyicao@foxmail.com
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/2 16:06   shuaiyicao    1.0         None
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from functools import reduce
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,KFold


def load_csv(file_path):
    df = pd.read_csv(file_path, index_col=False, encoding='utf-8')
    return df


def fea_process(df, drop_list, cat_fea):
    # init num_fea cat_fea
    num_feature = []
    cat_feature = []

    for i in list(df.columns):
        if i in drop_list:
            continue
        elif i in cat_fea:
            cat_feature.append(i)
        else:
            num_feature.append(i)
    for i in cat_feature:
        df[i] = df[i].astype('category')

    features = num_feature + cat_feature

    return features, num_feature, cat_feature


def train_pred_lgb(train_data, features, test_data, cat_feature, file_path):
    train_x = train_data[features]
    train_y = train_data['label']
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    y_pred = np.zeros(len(test_data))

    for fold, (train_index, val_index) in enumerate(kf.split(train_x, train_y)):
        x_train, x_val = train_x.iloc[train_index], train_x.iloc[val_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[val_index]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        params = {
            'boosting_type': 'gbdt',
            'metric': {'auc'},
            'objective': 'binary',
            'seed': 2020,
            'num_leaves': 50,
            'learning_rate': 0.1,
            'max_depth': 10,
            'n_estimators': 5000,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'colsample_bytree': 0.7,
            'verbose': 1,
        }

        model = lgb.train(params, train_set, categorical_feature=cat_feature, num_boost_round=5000,
                          early_stopping_rounds=50, valid_sets=[val_set], verbose_eval=100)
        y_pred += model.predict(test_data[features], num_iteration=model.best_iteration) / kf.n_splits

    print(y_pred)
    test_data['score'] = y_pred
    test_data[['id', 'score']].to_csv(file_path, index=False)


if __name__ == '__main__':
    # load fea data
    train_fea = load_csv('./fea/fea_1104.csv')

    # drop some features
    # drop_list = ['opto', 'opform', 'enttypeminu', 'adbusign', 'venind']
    # for col in drop_list:
    #     del train_fea[col]

    # cata fea and drop fea
    cat_fea = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
       'enttypeitem', 'state', 'orgid', 'jobid', 'adbusign', 'townsign',
       'regtype', 'compform', 'venind', 'enttypeminu', 'protype', 'enttypegb']
    drop = ['id', 'label']
    features, num_feature, cat_feature = fea_process(train_fea, drop, cat_fea)

    # load test data
    test_id = load_csv('../ccf_data/entprise_submit.csv')
    # base_info = load_csv('../ccf_data/base_info.csv')
    base_info = load_csv('../code/fea_explore/base_info.csv')
    other_info = load_csv('../code/fea_explore/other_info.csv')
    news_info = load_csv('../code/fea_explore/news_info.csv')
    dfs = [test_id, base_info, other_info, news_info]
    test_data = reduce(lambda left, right: pd.merge(left, right, on=['id'], how='left'), dfs)
    # test_data = pd.merge(test_id, base_info, on=['id'], how='left')

    # test_data['year'] = test_data['opfrom'].apply(lambda x: int(x.split('-')[0]))
    # test_data['month'] = test_data['opfrom'].apply(lambda x: int(x.split('-')[1]))
    # del test_data['opfrom']

    for i in cat_feature:
        test_data[i] = test_data[i].astype('category')

    fillna_col = ['news_sum', 'news_pos_sum', 'news_mid_sum', 'news_neg_sum']
    for col in fillna_col:
        test_data[col].fillna(0, inplace=True)

    test_data.to_csv('./test_data.csv', index=False)

    # trian and predict
    train_pred_lgb(train_fea, features, test_data, cat_feature, './submit_1105.csv')
