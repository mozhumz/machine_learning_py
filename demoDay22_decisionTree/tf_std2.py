import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

IDIR = 'G:\\bigdata\\badou\\00-data//'
df_train = pd.read_csv(IDIR + 'train_feat.csv').fillna(0.).astype(pd.SparseDtype("float", np.nan))
print(df_train)

labels = np.load(IDIR + 'labels.npy')
print(labels)

X_train, X_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2, random_state=2019)

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
        if shuffle:
            # batchsize = NUM_EXAMPLES 时，batchsize过大会报kernal restarting 错误
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = (dataset
                   .repeat(n_epochs)
                   .batch(NUM_EXAMPLES))
        return dataset

    return input_fn


# Training and evaluation input functions.
train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_test, y_test, shuffle=False, n_epochs=1)


f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
            'user_average_days_between_orders', 'user_average_basket',
            'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
            'aisle_id', 'department_id', 'product_orders', 'product_reorders',
            'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
            'UP_average_pos_in_cart', 'UP_orders_since_last',
            'UP_delta_hour_vs_last']
params = {
    'n_trees': 50,
    'max_depth': 3,
    'n_batches_per_layer': 1
}
est = tf.estimator.BoostedTreesRegressor(f_to_use, **params)
est.train(train_input_fn, max_steps=100)
results = est.evaluate(eval_input_fn)
df = pd.Series(results).to_frame()
print('df', df)
y_pred = est.predict(eval_input_fn)
print('auc_test1:', roc_auc_score(y_test, y_pred))
