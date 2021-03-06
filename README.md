IJCAI2018-比赛总结<br><br>
=======================

# 1 数据预处理
有10几个重复的数据样本，去重
Id、性别、年龄的缺失值填充的是-1,可以填充为nan(树模型可以处理)
对时间戳做处理，得到日期和时间<br><br>
# 2 特征工程<br>
## 2.1 基础特征
基础特征简要介绍下，主要获取用户、商品、店铺的特征，比如用户性别、年龄、职业、星级编号，商品类目、品牌、城市、价格等级、展示次数、搜藏次数等级，店铺评价数量等级、好评数量等级、服务态度评分等等
另外一些组合特征，比如用户当次搜索距离当天第一次搜索该商品、店铺、类目的时间差，用户当次搜索距离当天最后一次搜索该商品、店铺、类目的时间差
一些商品的相对特征，商品相对同类别、品牌的平均价格差，销量差<br>
## 2.2 组合特征
用户当日、当小时搜索总数
用户当日搜索该商品、店铺、类目次数
用户当小时搜索该商品、店铺、类目次数<br>
## 2.3 统计特征<br>
用户、商品、店铺、品牌、类目在这天之前的买入总数
用户、商品、店铺、品牌、类目前1、2、3、4天之前的转化率
商品的属性是否出现在预测属性中（商品属性是否符合用户要求）
商品属性中有多少个出现在预测属性中
商品属性对的转化率
用户、商品、店铺、品牌前30分钟、1、4、12个小时的浏览量，用户前4个小时的浏览量处于前24个小时的浏览量(相当于对浏览量做了归一化处理) <br><br>
# 3 模型
把特征读取进来后做如下处理
1、  对缺失值进行填充，填充成中位数
2、  对['item_sales_level','item_collected_level','item_pv_level']取一下对数
3、  做采样率为0.77的下采样
4、  截取最后一天的数据拿来做验证集，大概占训练集的11%
5、  生成一个onehot版本的数据用于LR模型(由于效果较差，后期没有用到)，对id类的特征做one-hot编码会使维度变得非常高，难以训练，所以id类的特征不做one-hot。对类别特征，比如性别，职业，用户、商品、店铺评级，商品类别、商品城市、商品品牌等做one-hot。
6、  对用户、商品、店铺、品牌、类别这5列的CVR拿过来做一个融合，融合成一列，因为这些CVR存在很多-1的数据，不利于模型的训练。融合后希望尽量避免-1的数据同时又比较接机原始数据。表现出的意思就是，只有当5列同时为-1，融合数据才为-1，否则不是。<br>
## 3.1 LGB
LGB采用gbdt的模型，树模型的深度为6，特征选择率为0.7，学习率为0.1,最大迭代次数为6000, early_stopping的轮数为500. 做5折的CV训练，每折都对测试集做一次预测，然后5列结果取中数或者是平均数。实际的迭代轮数差不多是350轮到400轮左右。线下的训练集logloss为0.7828，测试集为0.08641，线下验证集logloss为0.079.<br>
## 3.2 XGB
XGB模型采用深度为6，列采样为0.85，lambda为1，eta为0.4的参数来训练。最大轮数为500轮，early_stopping的轮数为7轮。做5折的CV训练，每折都对测试集做一次预测，然后5列结果取中数或者是平均数。实际的迭代轮数差不多是35轮到40轮左右。线下的训练集logloss为0.7013，测试集为0.08812，线下验证集logloss为0.07919.<br>
## 3.4 FFM
对于FFM模型还要做域的划分，我们把每个特征都看成是一个域，同时对连续值做分箱操作，然后结合类别特征做one-hot编码。最后扔到FFM模型最训练。FFM模型的参数如下：lr为0.05，lambda为0.0000005，epoch为70，alpha为0.1，lambda_1为0.01<br>
## 3.5 尝试过的方法和模型
分阶段训练，首先用对数据做一个粗预测，对预测结果从高到底排序后按比例分成两部分，比如取分数最高的前10%，我们认为这部分样本可能会购买，我们对它放到下一层模型进行训练和预测。而对于低90%的样本我们认为它很有可能是不够买的，我们也把放到另外的一层模型做训练。结果发现这样效果也不太好。
分新用户和老用户训练，经过特征重要度分析，我们发现模型比较依赖于用户的CVR，而对于新用户我们是无法知道它的CVR的，只能得到老用户的CVR。于是我们把用户分成新用户和老用户，分别使用不同的模型训练，然后拼接得到最终的结果。发现效果也是不佳。
GBDT+LR模型，这个模型被Facebook发表在一篇论文上，据说很厉害，在kaggle比赛上也得到了验证。LR的特征就是GBDT的叶节点编号，这个思想也是特征组合的思想。但是我们把这个模型用起来后结果发现，效果并不好，比LGB、XGB的效果都差很多，比赛的网友也说效果不好，不解。<br><br>
# 4 融合
融合试过中值融合、平均值融合，和Stacking融合(LR模型)，结果是中值好一点<br><br>
# 5 复赛总结
复赛的数据是一样的，但是有这两个不同的地方。第一，复赛的数据多达1000多万，对于特征的提取带来一些问题。第二，初赛的转化率差不多每天都一样，但是复赛不同需要预测特殊日期下的转化率，前7天数据是正常日期下的数据，而第8天开始搞促销，训练数据里面包含第7天上午的数据，我们需要预测第八天下午用户的转化率。
对于第一个问题，我们对于特征的提取采用多线程的办法，原始数据的大小是10多个G，内存小于16G的电脑可能读不进来。读进来后把数据分成64份，分别保存起来。在后面做特征的时候，创建64个进程，处理各自的数据，如果是和时间相关的数据(需要用到其他时间段的数据)，则需要把数据全部读出来做特征。
对于第二个问题，一开始的想法是只对第八天上午做训练，然后预测第八天下午数据，结果发现效果一般，同时丢失了前7天很多有用的信息。之后考虑了另一种方案，前7天只提取特征，然后把拿来的特征放到第八天做训练。比如提取的特征有：用户、店铺、商品、品牌、类别在前七天的购买量、浏览量、转化率，还可以有交叉特征，用户前七天浏览所有商品、店铺、类别、品牌的数量，店铺在前七天卖出商品、品牌、类别的数量等等。提交后发现这个效果还可以。
对模型的优化，只训练第七天，数据量也有限，为了更好的利用数据，可以使用全部的数据做训练，以当前天之前的数据作为特征提取的区间，这样的话，第一天是获取不到历史数据的，我们考虑将第1天的数据删除。
最后结果是0.14053，排名为100/5204<br><br>
# 6 其他参赛队
数据分析。分析到每天，每个小时的转化率。
特征。时间上的划分根据转化率来，可以看出晚上转化率略高(大家下班回家了)。店铺评分都差不多，但是分箱之后与转化率的相关性增强，同时利于做交叉特征。通过点击的特征提取出用户喜好。排序特征，比如销量、价格的排序，然后做归一化处理，排名/总数。Item的属性不要那么多，只列出常见的100多种属性拿来提取特征。统计每个人在属性上的平均值，类似于用户的属性喜好。统计item在这些属性上的点击率(商品的欢迎程度)。
模型。LGB、XGB、DeepFFM、FFM、LR。先对普通天做一个训练，然后预测第八天的数据，把这个预测数据作为一个新特征加入到第八天中(强特)。
融合。树模型先做一个加权，线性模型做一个加权，然后再做一个加权。加权融合、Stacking融合。
Trick。第七天的样本权重加大一点可以提升万分之一。对数据类型转换节省内存，int64变int8，float64变float16。
