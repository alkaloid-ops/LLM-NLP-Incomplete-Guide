
## LLaMA（Large Language Model Meta AI）

![IMG_0011.png](Large%20Language%20Models%20(LLM)%20&%20Neural%20Language%20Proc/IMG_0011.png)

### 词嵌入（Embedding）

- 输入数据X形状(batch_size, seq_len)，经过Embedding后尺寸为(batch_size, seq_len, d_model)
- 届时与输出时Linear层的权重参数共享

### 均方根归一化（RMSNorm）

- 对每个token的词向量求平方后的均值（不减均值-不作平移），再求均值的平方根（同时加上epsilon防止为零），计算出的这个数值为归一化缩放数值
- 与输入数据作归一化计算后，再乘一个可学习迭代参数gamma后（通常初始化为全1向量），完成均方根归一化（归一化计算方向仍是和LayerNorm一样）
- RMSNorm计算公式

$$
RMSNorm(x)=\frac{x}{\sqrt {\frac{1}{n}\sum^n_{i=1} x_i^2 +\epsilon}} *\gamma
$$

- 与Transformer不同的是，LLaMA始终先归一化（Pre-Norm）

### 旋转位置编码（RoPE）

- 均方根归一化后，对Q和K矩阵的每行向量从左往右两两一对（一偶数索引和一奇数索引）作为分量，与旋转矩阵做矩阵乘法，得到旋转后的向量（即旋转位置编码），后续无需再和输入数据求和
- 旋转矩阵：以超参数指定旋转角度，自动计算旋转矩阵具体数值
    
    $$
    二维绕原点逆时针旋转矩阵\begin{bmatrix}
     cos(\theta_i) & -sin(\theta_i) \\
     sin(\theta_i) & cos(\theta_i)
    \end{bmatrix}
    $$
    

### KV缓存

- 在解码器自回归生成token时，只计算当前一个token的注意力，随后输出的token会再次输入解码器（之前计算的KV参数保存作为KV缓存），此时解码器不再重复计算历史token的注意力，当前token注意力和历史token注意力以token序列顺序拼接成KV矩阵，再做注意力计算
- Q矩阵不缓存（始终根据当前token来计算）
- LLaMA初始架构全局没有偏置Bias参数和Dropout丢弃注意力和神经元，后续v2版本开始增加了偏置Bias

### 注意力衍生机制

- 多头注意力机制（MHA）：生成和头数一致的QKV参数矩阵
    - eg: 如8头注意力机制将从源QKV三矩阵分割生成8个Q矩阵、8个K矩阵、8个V矩阵，一共24个参数矩阵）
- 多查询注意力机制（MQA）：生成和头数一致的矩阵，K和V矩阵各一个，作为KV缓存共享计算使用
    - eg: 如8头注意力机制将从源QKV三矩阵分割生成8个Q矩阵、1个K矩阵、1个V矩阵，一共10个参数矩阵）
- 多组查询注意力（GQA）：生成和头数一致的Q矩阵，生成头数/组数的K矩阵和V矩阵，组内K矩阵和V矩阵复制数量以满足Q矩阵计算
    - eg: 如8头分2组注意力机制将从源QKV三矩阵分割生成8个Q矩阵，4个K矩阵和4个V矩阵，一共16个参数矩阵）
- 每个QKV子矩阵尺寸为(batch_size, heads, seq_len, d_model/heads)

### 门控激活函数

- GLU
    - 当数据输入神经网络输入层后，会分成两路（类似立体双隐藏层并行结构），得到两个计算值a和b，最终a和b逐元素相乘得到激活后的值
    - 门控机制和LSTM的门控机制原理一致，对输入数据信息选择性放行
    
    $$
    GLU(x)=a\odot sigmoid(b)=(xW_a+b_a)\odot sigmoid(xW_b+b_b)
    $$
    
- SwiGLU
    - 当数据输入神经网络输入层后，会分成两路（类似立体双隐藏层并行结构），得到两个计算值a和b，然后b通过Swish激活函数先行计算后，再和a作逐元素相乘得到最终激活后的值
    - 与GLU相比，在门控部分增加了缩放b（也就是对sigmoid的计算值进行缩放），而且不是一个固定单一的缩放因子，是根据数据线性映射后对每个特征（每个神经元）进行动态缩放（取决于各个特征的重要性），最终实现数据的非线形特征重组

$$
SwiGLU(x)=a\odot Swish(b)=a\odot (b\cdot sigmoid(b))=(xW_a+b_a)\odot( (xW_b+b_b) \cdot sigmoid(xW_b+b_b)))
$$

### 前馈神经网络（FNN）

- 与Transformer的FNN结构不同，LLaMA的FNN引入门控激活函数（SwiGLU），以及FNN的总层数为3层（输入层-隐藏层-输出层）

$$
FFN(x)=W_0(xW_a \odot ((xW_b)\cdot sigmoid(xW_b)))(其中W_a和W_b为隐藏层参数,W_0为输出层参数)
$$

- FNN输出和输入依旧残差连接Add（求和）

### 最终输出层

- 最终输出层的结构为RMSNorm→Linear→Softmax
- 计算出的注意力经过FNN进行非线形特征重组后，再进行RMSNorm均方根归一化后，输入Linear（仅一层线性层）作线性映射（这个层的参数和最初Embedding层的参数共享），再次映射到词汇表（也就是反向Embdding），得到词汇表各个候选词的计算值，最后经过Softmax转换为概率值