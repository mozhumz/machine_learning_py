# 推荐系统
## 一、任务执行顺序
1. `gen_cf_data.py`:生成训练数据，供CF使用和后续lr训练和在线部分获取用户属性信息使用
2. `cf_rec_list.py`:生成协同过滤召回（match）模块离线数据，供线上推荐系统召回时使用，一般数据需要存储到线上存储系统（redis）
3. `lr_train.py`：生成排序模块的模型，在lr模型中“模型”就是“w”和“b”
4. `rec_system.py`:模拟线上请求流程模块，当用户来了之后要经历线上的召回和排序模块，分别需要线上使用离线计算好的数据和模型
5. `config.py`:这个代码是作为数据的和一些参数的配置文件

python机器学习算法包`sk-learn`
注意： 在使用前，需要提前更改`config.py`中的数据路径，包括读入的数据路径和输出时候的数据路径
## 二、离线部分任务
## 三、在线部分任务
