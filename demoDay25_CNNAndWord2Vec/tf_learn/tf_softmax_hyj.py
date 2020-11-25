import demoDay21_recsys_music.hyj.config_hyj as conf
import pandas as pd
import demoDay21_recsys_music.hyj.gen_cf_data_hyj as gen
import tensorflow as tf
import numpy as np
data=gen.user_item_socre(nrows=500)
# 定义label stay_seconds/total_timelen>0.9 -> 1
data['label']=data['score'].apply(lambda x:1 if x>=0.9 else 0)


# 关联用户信息和item信息到data
user_profile=conf.user_profile()
music_meta=conf.music_data()

# data数据结构
# user_id	                            item_id	score	label	gender	age	salary	province	item_name	total_timelen	location	tags
# 0	0000066b1be6f28ad5e40d47b8d3e51c	426100349	1.280	1	女	26-35	10000-20000	香港	刘德华 - 回家的路 2015央视春晚 现场版	250	港台	-
# 1	000072fc29132acaf20168c589269e1c	426100349	1.276	1	女	36-45	5000-10000	湖北	刘德华 - 回家的路 2015央视春晚 现场版	250	港台	-
# 2	000074ec4874ab7d99d543d0ce419118	426100349	1.084	1	男	36-45	2000-5000	宁夏	刘德华 - 回家的路 2015央视春晚 现场版	250	港台	-
data=data.merge(user_profile,how='inner',on='user_id').merge(music_meta,how='inner',on='item_id')

''' 定义特征X'''

#用户特征
user_feat=['age','gender','salary','province']
# 物品特征
item_feat=['location','total_timelen']
item_text_feat=['item_name','tags']
# 交叉特征
watch_feat=['stay_seconds','score','hour']

# 离散和连续特征
dispersed_feat=user_feat+['location']
continue_feat=['score']

# 获取Y （label）
labels=data['label']
del data['label']

# 离散特征-one-hot处理
# df数据结构 ：
# age_0-18	age_19-25	age_26-35	age_36-45	age_46-100	gender_女	gender_男	salary_0-2000	salary_10000-20000	salary_2000-5000	...	province_香港	province_黑龙江	location_-	location_亚洲	location_国内	location_日韩	location_日韩,日本	location_日韩,韩国	location_欧美	location_港台
# 0	0	0	1	0	0	1	0	0	1	0	...	1	0	0	0	0	0	0	0	0	1
# 1	0	0	0	1	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	1
# 2	0	0	0	1	0	0	1	0	0	1	...	0	0	0	0	0	0	0	0	0	1
# 3	0	1	0	0	0	1	0	0	0	1	...	0	0	0	0	0	0	0	0	0	1
# 4	0	0	0	1	0	0	1	0	1	0	...	0	0	0	0	0	0	0	0	0	1
# get_dummies one-hot处理: 将每列展开成多列,有值的为1，否则为0(根据每列的所有取值,如用户性别男 gender取值有男女 则展开为gender_女 0	gender_男 1)
df=pd.get_dummies(data[dispersed_feat])
# 离散特征数组
# one-hot数据结构
# Index(['age_0-18', 'age_19-25', 'age_26-35', 'age_36-45', 'age_46-100',
#        'gender_女', 'gender_男', 'salary_0-2000', 'salary_10000-20000',
#        'salary_2000-5000', 'salary_20000-100000', 'salary_5000-10000',
#        'province_上海', 'province_云南', 'province_内蒙古', 'province_北京',
#        'province_台湾', 'province_吉林', 'province_四川', 'province_天津',
#        'province_宁夏', 'province_安徽', 'province_山东', 'province_山西',
#        'province_广东', 'province_广西', 'province_新疆', 'province_江苏',
#        'province_江西', 'province_河北', 'province_河南', 'province_浙江',
#        'province_海南', 'province_湖北', 'province_湖南', 'province_澳门',
#        'province_甘肃', 'province_福建', 'province_西藏', 'province_贵州',
#        'province_辽宁', 'province_重庆', 'province_陕西', 'province_青海',
#        'province_香港', 'province_黑龙江', 'location_-', 'location_亚洲',
#        'location_国内', 'location_日韩', 'location_日韩,日本', 'location_日韩,韩国',
#        'location_欧美', 'location_港台'],
#       dtype='object')
one_hot_cols=df.columns
# 连续特征不做one-hot 直接存储
df[continue_feat]=data[continue_feat]
df['label']=labels
pre='F:\\八斗学院\\视频\\14期正式课\\00-data\\nn\\'
df_file=pre+'music_data.csv'
df.to_csv(df_file,index=False)
print('df to csv done')
chunks = pd.read_csv(df_file,iterator = True,chunksize=1000)
chunk = chunks.get_chunk(5)
print(chunk)
print(np.array(chunk))
# 定义超参数
sample_num=len(df)
x_col_num=len(df.columns)-1
y_class_num=2
x=tf.placeholder(dtype=tf.float32,shape=[None,x_col_num],name='X')
y=tf.placeholder(dtype=tf.float32,shape=[None,y_class_num],name='Y')
max_epoches=10000
learning_rate=0.01
batch_size=50
seed=0
n_hidden=30

del df
# 定义权重和偏置变量
w={'h1':tf.Variable(tf.random_normal([x_col_num,n_hidden],seed=seed)),
   'res':tf.Variable(tf.random_normal([n_hidden,y_class_num],seed=seed))
   }

b={'h1':tf.Variable(tf.random_normal([1,n_hidden],seed=seed)),
   'res':tf.Variable(tf.random_normal([1,y_class_num],seed=seed))
   }
# 创建模型
def multilayer(x,w,b):
    # 隐藏层net值
    h_1=tf.add(tf.matmul(x,w['h1']),b['h1'])
    # 隐藏层out值
    h1_out=tf.sigmoid(h_1)
    # 输出层net值
    res_net=tf.add(tf.matmul(h1_out,w['res']),b['res'])
    return res_net,res_net,h1_out,h_1
# 为正向传播、误差、梯度和更新计算创建计算图
y_hat,y_net,h_out,h_net=multilayer(x,w,b)

'''
# softmax_cross_entropy_with_logits 表示先求softmax，再求交叉熵 等价于
loss2 = (-tf.reduce_sum(y*tf.log(tf.clip_by_value(softmax(y_hat),1e-10,1.0))))
clip_by_value防止y_hat取对数为0
'''
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))

# 梯度优化器
optimizer=tf.train.AdamOptimizer().minimize(loss)
# 变量初始化器
init=tf.global_variables_initializer()

# 读取数据
def data_gen(filename):
    f_queue=tf.train.string_input_producer(filename)
    reader=tf.TextLineReader(skip_header_lines=1)
    _,value=reader.read(f_queue)
    record_defaults=[[0.0] for _ in range(x_col_num+1)]
    data=tf.decode_csv(value,record_defaults=record_defaults)
    print(type(data))
    feats=tf.stack(tf.gather_nd(data,indices=[[i] for i in range(x_col_num)]))
    label=data[-1]
    dad=10*batch_size
    cap=20*batch_size
    feat_batch,label_batch=tf.train.shuffle_batch([feats,label],batch_size=batch_size,min_after_dequeue=dad,capacity=cap)

    return feat_batch,label_batch

def gen_data(feat_batch,label_batch):
    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        feats,labels=sess.run([feat_batch,label_batch])
        # for _ in range(5):
        coord.request_stop()
        coord.join(threads)
        return feats,labels

auc_bool_arr=tf.equal(tf.argmax(y,1),tf.argmax(y_hat,1))
auc=tf.reduce_mean(tf.cast(auc_bool_arr,tf.float32))
# 保存模型
saver = tf.train.Saver()


def train():
    for i in range(5):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 每次训练batch_size的样本量
        batch_xs, batch_ys = data_gen([df_file])
        train_auc(batch_xs, batch_ys)
        coord.request_stop()
        coord.join(threads)
    # try:
    #     while not coord.should_stop():
    #
    # except tf.errors.OutOfRangeError:
    #     print('read done')
    # finally:
    #     pass
    # Wait for threads to finish.



def train_auc(batch_xs, batch_ys):
    print(batch_xs.shape)
    batch_xs_val, batch_ys_val = sess.run([batch_xs, batch_ys])
    print(batch_xs_val[-1], batch_ys_val[-1])
    batch_ys2 = []
    for j in batch_ys_val:
        if j < 1.0:
            batch_ys2.append([1., 0.])
        else:
            batch_ys2.append([0., 1.])
    y_hat_out, _, loss_val, auc_val = sess.run([y_hat, optimizer, loss, auc],
                                               feed_dict={x: batch_xs_val, y: batch_ys2})
    print('loss %s,auc %s' % (loss_val, auc_val))


with tf.Session() as sess:
    sess.run(init)
    # writer=tf.summary.FileWriter('graphs',graph=sess.graph)
    for epoch in range(20):
        loss_avg=0.
        loss_avg2=0.
        # 总的批次数
        num_of_batch=int(sample_num/batch_size)
        print('epoch %s'%epoch)
        train()
    saver.save(sess,save_path=pre+'nn_softmax_model.ckpt')

    print('main done')

