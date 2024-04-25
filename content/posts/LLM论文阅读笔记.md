---
title: "LLM论文阅读笔记" 
date: 2024-04-25T17:09:46+08:00 
lastmod: 2024-04-25T17:09:46+08:00 
author: ["张广益"] 
tags: ["LLM"]
description: "大模型相关论文阅读笔记" #描述
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

## Phi-3

论文地址：https://arxiv.org/pdf/2404.14219

模型地址：https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3

代码地址：暂无

这篇论文的标题是《Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone》，由Microsoft团队撰写。论文介绍了一个名为phi-3-mini的新型语言模型，它包含38亿参数，并且是在3.3万亿token上训练得到的。phi-3-mini的性能在学术基准测试和内部测试中与Mixtral 8x7B和GPT-3.5等模型相当，尽管它的大小足以部署在手机上。这一成就完全归功于训练数据集的创新，该数据集是phi-2所用数据集的扩展版本，包含了大量过滤的网络数据和合成数据。此外，该模型还针对鲁棒性、安全性和聊天格式进行了进一步的优化。
主要内容包括：

1. **介绍**：论文讨论了大型语言模型（LLMs）的发展历程，以及通过使用高质量的训练数据来提高小型语言模型性能的方法。

2. **技术规格**：phi-3-mini模型采用了变换器解码器架构，具有4K的默认上下文长度，并通过LongRope技术扩展到了128K的上下文长度。模型基于Llama-2的相似块结构构建，并使用了相同的分词器，词汇量为320641。

3. **训练方法**：phi-3-mini的训练分为两个阶段：第一阶段主要使用网络数据教授模型通用知识和语言理解；第二阶段结合了更严格的网络数据和一些合成数据，教授模型逻辑推理和各种专业技能。

4. **数据最优范围**：与以往在“计算最优范围”或“过度训练范围”训练语言模型的工作不同，本研究主要关注给定规模下的数据质量。

5. **后期训练**：phi-3-mini的后期训练包括监督式微调（SFT）和直接偏好优化（DPO），使用了高质量的数据进行微调，并针对聊天格式数据、推理和负责任的AI（RAI）进行了优化。

6. **学术基准测试**：论文报告了phi-3-mini在标准开源基准测试上的结果，这些测试衡量了模型的推理能力，包括常识推理和逻辑推理。

7. **安全性**：phi-3-mini的开发遵循了Microsoft的负责任AI原则，包括在后期训练中的安全性对齐、红队测试、自动化测试和评估。

8. **局限性**：尽管phi-3-mini在语言理解和推理能力上与更大的模型相当，但由于其大小限制，它在某些任务上的能力仍然有限，例如在TriviaQA上的性能较低。然而，通过与搜索引擎结合，可以解决这一弱点。

9. **参考文献**：列出了与本研究相关的文献。

10. **附录**：提供了用于基准测试的示例提示。

论文的亮点在于展示了即使在小型设备上也能部署具有高性能的语言模型，这对于推动语言模型在移动设备上的应用具有重要意义。此外，通过精心策划和优化训练数据集，研究人员能够在不牺牲性能的情况下显著减少模型的大小。

```python
from llama_cpp import Llama


llm = Llama(
  model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
  n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8, # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=35, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)

prompt = "How to explain Internet to a medieval knight?"

# Simple inference example
output = llm(
  f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
  max_tokens=256,  # Generate up to 256 tokens
  stop=["<|end|>"], 
  echo=True,  # Whether to echo the prompt
)

print(output['choices'][0]['text'])

```

$$
l_i = -\log \frac{e^{sim(h_i^{z_i}, h_i^{z_i'})/ \tau}}{\sum_{j=1}^N e^{sim(h_i^{z_i}, h_j^{z_j'})/ \tau}}
$$

![phi3](phi-3.png)

