'''权重自动更新'''
import pandas as pd
import numpy as np



def get_wikiid_score(wkiid_dict):
    """
    输入用户队列的所有任务的信息，输出最先处理的一个任务
    :param wkiid_dict: wkiid的一个字典    :return:wkiid和对应分值的一个字典
    """
    df = pd.DataFrame.from_dict(wkiid_dict, orient='index',
                                columns=['match_rate', 'is_nine_trs', 't1_distinct', 't2_distinct', 't3_distinct',
                                         't4_distinct',
                                         'is_corr_nod', 'strorg', 'trschl', 'ntsgrp', 'cusflg',
                                         'paynum', 'drenum', 'imgcnt', 'amount', 'cnyno', 'inx_roles', 'inx_grpid',
                                         'yichu_priori',
                                         'fstflg'])
    wkiid_scores = {}

    def getNormalization(min, max):
        def min_max_scaler(num):
            num = round(num, 3)
            if max == 0:
                return 0
            elif max == min:
                return 0.1
            else:
                return float((num - min) / (max - min))

        return min_max_scaler

    t1_range = getNormalization(df['t1_distinct'].min(), df['t1_distinct'].max())
    t2_range = getNormalization(df['t2_distinct'].min(), df['t2_distinct'].max())
    t3_range = getNormalization(df['t3_distinct'].min(), df['t3_distinct'].max())
    t4_range = getNormalization(df['t4_distinct'].min(), df['t4_distinct'].max())
    acc_range = getNormalization(df['amount'].min(), df['amount'].max())
    rol_range = getNormalization(df['inx_roles'].min(), df['inx_roles'].max())
    grp_range = getNormalization(df['inx_grpid'].min(), df['inx_grpid'].max())
    for key, value in wkiid_dict.items():
        match_rate, is_nine_trs, t1_distinct, t2_distinct, t3_distinct, t4_distinct, is_corr_nod, strorg, trschl, ntsgrp, cusflg, paynum, drenum, imgcnt, amount, cnyno, inx_roles, inx_grpid, yichu_priori, fstflg = value
        if match_rate == 100:
            wkiid_scores[key] = 1  # 战略、绿色、立等 这些条件下这个任务对应的高优分值为1
        elif match_rate == 0:
            wkiid_scores[key] = 0  # 不可匹配的情况绝对不能返回该节点
        else:
            score = float(0.0) * match_rate + float(0.2) * is_nine_trs + \
                    float(0.15) * round(t1_range(t1_distinct), 5) + \
                    float(0.1) * round(t2_range(t2_distinct), 5) + float(0.06) * round(t3_range(t3_distinct), 5) + \
                    float(0.04) * round(t4_range(t4_distinct), 5) + float(0.15) * round(is_corr_nod, 5) + \
                    float(0.05) * round(acc_range(amount), 5) + float(0.05) * round(rol_range(inx_roles), 5) + \
                    float(0.05) * round(grp_range(inx_grpid), 5) + \
                    float(0.1) * yichu_priori + \
                    float(0.05) * fstflg
            wkiid_scores[key] = score
    return wkiid_scores


