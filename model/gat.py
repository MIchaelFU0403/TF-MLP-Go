from __future__ import absolute_import


# https://blog.csdn.net/weixin_43460251/article/details/106480331?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-106480331-blog-102782687.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-106480331-blog-102782687.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=4

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU

# 本文件中的代码最主要看的是__call__文件，展示的淋漓尽致

class GraphAttention(Layer):

    def __init__(self,  # 构造函数与初始化，带自身self参数的各个含义在下面赋值有阐述
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,  # dropout机制，主要是针对数据集较少的数据同时要求的参数又比较多，这样的数据往往会产生过拟合的现象，该机制可以防止相当于正则化。
                 activation='relu',  # 激活函数relu
                 use_bias=True,  # 偏移量
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper) 在论文里面的F`即输出的特征，加权求和得到的特征
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper) 是multi-heads的数量，即K为K轮
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        # 接上：multi-heads的结合方式，5为链接（concat）6为取均值（average）
        self.dropout_rate = dropout_rate  # Internal dropout rate dropout的比率，即不采取样本的多少值。
        self.activation = activations.get(activation)  # Eq. 4 in the paper 加权求和的公式
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)  # 权值初始化的方法
        self.bias_initializer = initializers.get(bias_initializer)  # 初始化偏移量
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)  # 初始化multi-heads

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads  # 如果输出为concat形式的特征那么输出就要扩大
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_  # 否则不变，即取均值

        super(GraphAttention, self).__init__(**kwargs)

    # 由于是第一次看代码，我在这里做一点科普。对于学习层的建造，是首先调用init初始函数然后在call使用之前调用一次build函数，如果是第二次使用就不再build了
    # 如果是使用已存在的layer是不需要写build函数的（只需self.built = true），只有自定义时才需要。
    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):  # 最核心的代码逻辑在这里
        X = inputs[0]  # Node features (N x F) 输入为X个点的F个特征值组成的矩阵
        A = inputs[1]  # Adjacency matrix (N x N) 输入为邻接矩阵

        outputs = []  # 定义输出元组
        for head in range(self.attn_heads):  # 对于我们需要的multi轮数之下进行不断地迭代
            kernel = self.kernels[head]  # W in the paper (F x F') # 我们一开始学习的扩展矩阵W，即将F个特征扩大维度到F`
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1) 映射向量，即对WH操作关联值

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F') 获得加权求和之前的输入特征矩阵，即对X和kernel进行矩阵的乘积

            # Compute feature combinations 注意！代码和论文的区别在这里！我尚不太懂这么操作的理由是什么
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]] 这里即计算出关系系数
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty ·定义·添加非线性函数作为激活函数
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017) 先做好mask。我们采用的是mask_attention
            mask = -10e9 * (1.0 - A)  # 如果点之间有变相连那么就取0，没有在下一步就会变成-inf相当于
            dense += mask

            # Apply softmax to get attention coefficients 使用激活函数
            dense = K.softmax(dense)  # (N x N)

            # 现在进入的是第二步！即加权求和部分，需要将特征进行加权求和
            # Apply dropout to features and attention coefficients，这里就是对特征和权值进行dropout
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            # 加权求和步骤
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)  # 将获得的特征F`加入到输出元组

        # 从这里开始是对multi-heads进行操作，即如果是链接的就把···在之前说过了 不赘述
        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

