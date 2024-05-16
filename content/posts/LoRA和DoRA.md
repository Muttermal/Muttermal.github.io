---
title: "LoRA和DoRA" 
date: 2024-05-16T16:17:45+08:00 
lastmod: 2024-05-16T16:17:45+08:00 
author: ["张广益"] 
tags: ["LLM"]
description: "简要介绍LLM微调方式LoRA&DoRA" #描述
weight: # 输入1可以顶置文章，用来给文章展示排序，不填就默认按时间排序
slug: ""
math: true	# 显示公式
draft: false # 是否为草稿
comments: false #是否展示评论
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: false # 底部不显示分享栏
showbreadcrumbs: true #顶部显示当前路径
cover:
  image: "" #图片路径
  caption: "" #图片描述
  alt: ""
  relative: false
---

# LoRA & DoRA

## LoRA

LoRA([Low-Rank Adaptation](https://arxiv.org/abs/2106.09685))，是一种利用低秩分解来降低LLM微调成本的技术。不同于传统的微调方式需要更新整个模型的参数，LoRA仅更新一小部分低秩矩阵的参数来达到类似的效果。
![LoRA](lora.png)

对于预训练的权重矩阵$W \in \mathbb{R}^{d \times  k}$，LoRA将模型的更新分为两部分：$$W' = W + \Delta W = W + \underline{BA}$$
对于更新部分的参数 $\Delta W$，LoRA 通过低秩分解将其分解成矩阵 $A$ 和矩阵 $B$，其中$B \in  \mathbb{R}^{d \times  r}, A \in  \mathbb{R}^{r \times  k}$。 $r$ 是一个超参数，我们可以用它来指定一个合适的低秩矩阵。较小的 $r$ 会导致更简单的低秩矩阵，从而导致在适应过程中需要学习的参数更少。这可以带来更快的训练并可能减少计算要求。然而，随着 $r$ 的减小，低秩矩阵捕获特定任务信息的能力会降低。在初始化时，矩阵 $A$ 为零矩阵，$B$ 为随机的高斯分布。


```python
import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
```

在上述代码中，我们还引入了一个缩放因子 $\alpha$，这个参数决定了LoRA 层对模型现有的权重改变的幅度：$\alpha * (x * A * B)$。$\alpha$越大，意味着对模型参数的改动幅度就越大，反之，模型的改动就越小。

通常LoRA层常应用于模型的**线性层**，如：

```python
# 正常的forward
def forward(self, x):
    x = self.linear_1(x)
    x = F.relu(x)
    x = self.linear_2(x)
    return x

# 加入LoRA层后的forward
def forward(self, x):
    x = self.linear_1(x) + self.lora_1(x)
    x = F.relu(x)
    x = self.linear_2(x) + self.lora_2(x)
    return logits
```

在训练时，我们可以通过修改模型的一些线性层来实现LoRA微调：


```python
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

我们以DistilBERT为例，完整的代码示例如下。


```python
from functools import partial
from transformers import AutoModelForSequenceClassification


# default hyperparameter choices
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

layers = []
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    if lora_query:
        layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = assign_lora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = assign_lora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
        layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = assign_lora(model.pre_classifier)
    model.classifier = assign_lora(model.classifier)

# Check if linear layers are frozen
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Total number of trainable parameters:", count_parameters(model))
```

增加LoRA层后，模型的原始参数被冻结，训练时只更新LinearWithLoRA层的参数。

## DoRA

DoRA([Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353))将预训练模型的权重分解为**大小magnitude**和**方向direction**两部分，然后微调magnitude部分，并使用LoRA对direction矩阵进行微调，从而提升模型的微调效果。DoRA 的论文结果表明在各种任务上效果均超越了 LoRA。

![DoRA](dora.png)

在DoRA的论文中，作者将模型的参数做了如下分解： 
$$
W = ||W||_{c} \frac{W}{||W||_{c}} = m \frac{V}{||V||_{c}} 
$$


其中$ m \in \mathbb{R}^{1 \times k} $ 为表示大小的向量，$ V \in \mathbb{R}^{d \times k} $为表示方向的单位矩阵。模型微调时变化的参数$\Delta W$也分解为大小上的变化$\Delta M$和方向上的变化$\Delta D$，分解后作者发现使用全参数微调时，$\Delta M$和$\Delta D$是成负相关的。DoRA的结果与全参数微调类似，而LoRA则是成正相关。

![DoRA](dora_2.png)

在实际训练时，可以使用模型的权重$W_0$来初始化DoRA，其中$m = ||W_0||_c \quad V=W_0$，然后更新$m$，同时使用LoRA来更新$V$：
$$W'= \underline{m} \frac{V+\Delta V}{||V + \Delta V||_c} = \underline{m} \frac{W_0+\underline{BA}}{||W_0 + \underline{BA}||_c}$$
下划线参数表示需要参与训练的参数，矩阵$A,B$以Lora的方式来初始化。以distilbert模型为例，示例代码如下：


```python
from functools import partial
from transformers import AutoModelForSequenceClassification
import torch


# default hyperparameter choices
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


class DoRALayer(nn.Module):
    def __init__(self, linear, rank=4, alpha=8):
        super().__init__()

        self.weight = nn.Parameter(linear.weight, requires_grad=False)
        self.bias = nn.Parameter(linear.bias, requires_grad=False)
        # m = Magnitude column-wise across output dimension
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        self.alpha = alpha
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(linear.out_features, rank)*std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.in_features))

    def forward(self, x):
        lora = self.alpha * torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)


layers = []
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False

assign_dora = partial(DoRALayer, rank=lora_r, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    if lora_query:
        layer.attention.q_lin = assign_dora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = assign_dora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = assign_dora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = assign_dora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = assign_dora(layer.ffn.lin1)
        layer.ffn.lin2 = assign_dora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = assign_dora(model.pre_classifier)
    model.classifier = assign_dora(model.classifier)

# Check if linear layers are frozen
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Total number of trainable parameters:", count_parameters(model))
```

---

参考链接：

1. [Code LoRA from Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?continueFlag=a210daa9eb709f41b7a1629398f95601&tab=overview&layout=column&path=cloudspaces%2F01hm9hypqc6y1hrapb5prmtz0h&y=7&x=0)
2. https://github.com/NVlabs/DoRA/tree/main
3. https://github.com/catid/dora/tree/main
