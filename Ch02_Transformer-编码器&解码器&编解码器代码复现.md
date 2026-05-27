
### 代码复现（Encoder-only）

```python
# Pytorch

# 位置编码
class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        
        # 构建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成 0-max_len的浮点数，并升维
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 计算位置编码的分母
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 正余弦函数计算位置编码
        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)
        # 位置编码二维矩阵增加维度（索引0的位置增维）为三维张量 (batch_size, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 注册为类属性
        self.register_buffer('pe', pe)

    def forward(self, x):
                        #只取seq_len长度的位置编码（seq_len＜＝max_len）
        return x + self.pe[:, :x.size(1), :]
        
# 文本分类器 (Encoder-only)
class TextClassifier(nn.Module):
    def __init__(self,
                vocab_size, #词汇表大小
                max_len,    #最大序列长度
                d_model,    #token维度
                n_head,     #头数
                n_layer,    #层数
                n_class,    #分类数
                d_ffn,      #前馈神经网络维度
                dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(max_len=max_len, d_model=d_model)

        # 构建编码器内部结构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        # 构建编码器及其数量
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
        )

        # 构建前馈神经网络（一层）
        self.fc = nn.Linear(d_model, n_class)
        self.dropout = nn.Dropout(dropout)

        # 将embedding后的缩放因子注册为类属性
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32)))

    def forward(self, x, padding_mask=None):
        
        # 输入x尺寸为(batch_size, seq_len)
        x = self.embedding(x)

        # Embedding后缩放，embedding信息突出把后续的位置编码信息淹没
        x = x * self.scale
        
        # Embedding后x尺寸为(batch_size, seq_len, d_model)
        x = self.position_encoding(x)

        # 增加位置信息后x尺寸不变，输入编码器
        output = self.encoder(x, src_key_padding_mask=padding_mask)
            
        # 聚合维度信息, 将序列长度聚合成一个时间步
        # 输出output尺寸变成（batch_size, d_model）
        # output = output.mean(dim=1) # 对每个batch中的每一维特征计算均值
        # output = output[:,0,:] # 直接取每个batch的第一个特殊标记CLS的token的特征（使用于长序列）
        
        if padding_mask is not None:
            # padding_mask是二维布尔值张量（batch_size, seq_len）
            # 将原本True=1，False=0通过波浪符取反True=0，False=1，并转换为浮点数，增加到第三维
            mask_expanded = (~padding_mask).unsqueeze(-1).float() 
            # 将编码器输出的结果中，padding部分的特征值置为0（外积计算）
            output = output * mask_expanded
            # 由于有0的存在，不能直接计算均值
            # 对每个特征求和
            sum_output = torch.sum(output, dim=1)
            # 计算False=1的数量，以及为防止极端情况中有某一batch中全部padding为0时，将0替换为1e-9，防止均值计算分母为0
            count_output = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            # 计算均值
            pooled_output = sum_output / count_output
        else:
            pooled_output = output.mean(dim=1)
            
        # 输入前馈神经网络，输出分类
        output = self.fc(self.dropout(pooled_output))

        return F.softmax(output, dim=1)

# Huggingface

pretrained_model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name,
    num_labels = 2
).to(device)

text_classifier = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    framework='pt',
    truncation=True,
    max_length=512,
    device=device
)

result = text_classifier(text) #传入的文本数据需要为列表格式
```

### 代码复现（Decoder-only）

```python
# Pytorch

# 位置编码
class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        
        # 构建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成 0-max_len的浮点数，并升维
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 计算位置编码的分母
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 正余弦函数计算位置编码
        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)
        # 位置编码二维矩阵增加维度（索引0的位置增维）为三维张量 (batch_size, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 注册为类属性
        self.register_buffer('pe', pe)

    def forward(self, x):
                        #只取seq_len长度的位置编码（seq_len＜＝max_len）
        return x + self.pe[:, :x.size(1), :]
        
# 文本生成器
class TextGenerator(nn.Module):
    def __init__(self,
                vocab_size, #词汇表大小
                max_len,    #最大序列长度
                d_model,    #token维度
                n_head,     #头数
                n_layer,    #层数
                d_ffn,      #前馈神经网络维度
                dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(max_len=max_len, d_model=d_model)

        # 将embedding后的缩放因子注册为类属性
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32)))
        
        # 构建编码器内部结构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # 构建编码器及其堆叠
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # 构建前馈神经网络（一层）
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # 因果掩码
    def CausalMask(self, seq_len, device):
        # 生成掩码矩阵，下三角矩阵，对角线往右偏移1个单位
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x, padding_mask=None):

        # embedding嵌入 + 缩放
        x = self.embedding(x) * self.scale

        # 增加位置编码
        x = self.position_encoding(x)

        # 生成因果掩码
        cm = self.CausalMask(seq_len=x.size(1), device=x.device)

        # GPT实际方式：
        output = self.decoder(x, mask=cm, src_key_padding_mask=padding_mask)

        # 前馈神经网络
        output = self.fc(self.dropout(output))

        return output

    def generate(self,
                 input_ids, #输入分词后的文本数据
                 max_new_tokens, #设置最大生成token数量
                 eos_token_id=None #
                ):
        # 推理模式：关闭dropout + batch_norm
        self.eval()
        # 关闭梯度
        with torch.no_grad():
            # 初始化生成token_id（一开始只能看到两个token）
            generated = input_ids
            for _ in range(max_new_tokens):
                output = self.forward(generated) #前向传播生成token
                next_token_output = output[:,-1,:] #取最后一个token的vocab
                next_token = next_token_output.argmax(dim=-1, keepdim=True) #取最后一个token的vocab的最大概率的token
                generated = torch.cat([generated, next_token], dim=1) #追加到生成序列中，相当于列表append
                if eos_token_id is not None:
                    if (next_token == eos_token_id).all(): #如果遇到eos就停止生成
                        break
        return generated
        
# Huggingface

pretrained_model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
inputs = tokenizer(prompt, return_tensors='pt').to(device)
output = model.generate(
        input_ids=inputs.input_ids,
        max_length=50,
        do_sample=True, # 启用采样（使结果更多样）
        temperature=1, #温度参数
        top_k = 50,
        top_p = 0.5,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

### 代码复现（Encoder-Decoder）

```python
# Pytorch

# 位置编码
class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        
        # 构建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成 0-max_len的浮点数，并升维
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 计算位置编码的分母
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 正余弦函数计算位置编码
        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)
        # 位置编码二维矩阵增加维度（索引0的位置增维）为三维张量 (batch_size, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 注册为类属性
        self.register_buffer('pe', pe)

    def forward(self, x):
                        #只取seq_len长度的位置编码（seq_len＜＝max_len）
        return x + self.pe[:, :x.size(1), :]
        
class MachineTranslation(nn.Module):
    def __init__(self,
                src_vocab_size, # 源语言词汇表
                tgt_vocab_size, # 目标语言词汇表
                max_len,        # 最大序列长度
                d_model,        # 语义特征维度
                n_head,         # 头数
                n_encoder_layer,# 编码器层数
                n_decoder_layer,# 解码器层数
                d_ffn,          # 前馈神经网络维度
                dropout=0.1):
        super().__init__()

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 生成位置编码矩阵
        self.pos_encoding = PositionEncoding(max_len, d_model)

        # 编码器与解码器参数
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_encoder_layer,
            num_decoder_layers=n_decoder_layer,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        # 构建前馈神经网络（一层）
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # 将embedding后的缩放因子注册为类属性
        self.register_buffer('scale', torch.sqrt(torch.tensor(d_model, dtype=torch.float32)))

    # 因果掩码
    def CausalMask(self, seq_len, device):
        # 生成掩码矩阵，下三角矩阵，对角线往右偏移1个单位
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
               src_key_padding_mask=None,
               tgt_key_padding_mask=None,
               memory_key_padding_mask=None):

        # 源语言词嵌入+缩放+位置编码
        src_embedding = self.src_embedding(src) * self.scale
        src_embedding = self.pos_encoding(src_embedding)

        # 目标语言词嵌入+缩放+位置编码
        tgt_embedding = self.tgt_embedding(tgt) * self.scale
        tgt_embedding = self.pos_encoding(tgt_embedding)

        if tgt_mask is None:
            tgt_mask = self.CausalMask(tgt.size(1), tgt.device)

        output = self.transformer(src_embedding, # 源语言词嵌入
                                  tgt_embedding, # 目标语言词嵌入
                                  src_mask=src_mask, # 编码器掩码（非必需）
                                  tgt_mask=tgt_mask, # 解码器掩码（必需）
                                  # 分词时的padding填充掩码
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  # 编码器的输出的padding掩码
                                  memory_key_padding_mask=memory_key_padding_mask)

        output = self.fc(self.dropout(output))

        return output

    def generate(self, src, max_new_tokens, bos_token_id, eos_token_id):
        # 推理模式：关闭dropout + batch_norm
        self.eval()
        # 关闭梯度
        with torch.no_grad():
            # 初始化解码输入：仅包含起始符
            generated = torch.tensor([[bos_token_id]], device=src.device)
            for _ in range(max_new_tokens):
                # 将当前已生成序列作为目标输入，得到模型输出
                output = self.forward(src, generated)
                # 取最后一个时间步的 logits，并取最大概率对应的 token
                next_token = output[:,-1,:].argmax(-1, keepdim=True)
                # 将新 token 拼接到已生成序列后
                generated = torch.cat([generated, next_token], dim=1)
                # 如果所有批次都生成了结束符，则提前终止
                if (next_token == eos_token_id).all():
                    break
        return generated
        
# Huggingface

pretrained_model_name = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
source_sentence = f'It was the best of times, it was the worst of times.'
inputs = tokenizer(
    text=source_sentence,
    padding=True,
    truncation=True,
    return_tensors='pt'
).to(device)
generated_ids = model.generate(
    inputs['input_ids'],
    max_new_tokens=50,
    num_beams=4,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
).to(device)
translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```
