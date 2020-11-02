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
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold


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


def get_predict(model, test_data, features, file_path):
    score = model.predict(test_data[features])
    test_data['score'] = score
    test_data[['id', 'score']].to_csv(file_path, index=False)


if __name__ == '__main__':
    # load fea data
    train_fea = load_csv('./fea/fea.csv')

    # drop some features
    drop_list = ['opto', 'opform', 'enttypeminu', 'adbusign', 'venind']
    for col in drop_list:
        del train_fea[col]

    # cata fea and drop fea
    cat_fea = ['industryphy', 'oploc', 'orgid', 'jobid']
    drop = ['id', 'label']
    features, num_feature, cat_feature = fea_process(train_fea, drop, cat_fea)

    lgb_model = lgb.LGBMRegressor(num_leaves=64, reg_alpha=0., reg_lambda=0.01, metric='rmse',
                                  max_depth=-1, learning_rate=0.05, min_child_samples=10, seed=2020,
                                  n_estimators=2000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1)
    train_x = train_fea[features]
    train_y = train_fea['label']
    lgb_model.fit(train_x, train_y)

    # load test data
    test_id = load_csv('../ccf_data/entprise_submit.csv')
    base_info = load_csv('../ccf_data/base_info.csv')
    test_data = pd.merge(test_id, base_info, on=['id'], how='left')

    test_data['year'] = test_data['opfrom'].apply(lambda x: int(x.split('-')[0]))
    test_data['month'] = test_data['opfrom'].apply(lambda x: int(x.split('-')[1]))
    del test_data['opfrom']

    for i in cat_feature:
        test_data[i] = test_data[i].astype('category')

    # predict
    get_predict(lgb_model, test_data, features, './submit_1102.csv')
