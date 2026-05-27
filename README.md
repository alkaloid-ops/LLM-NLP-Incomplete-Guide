# 📘 大语言模型（LLM）与自然语言处理（NLP）不完全指南

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

> 解析大语言模型底层架构、Transformer 核心机制、LLaMA 优化技术及 NLP 基础算法。公式推导 + PyTorch 代码实现，从零构建自己的 GPT / BERT / T5。

## 📌 简介

本仓库是本人整理的一份 **大语言模型与自然语言处理** 学习笔记，聚焦于 **Transformer 架构**、**LLaMA 系列模型** 以及 **NLP 经典算法**。内容涵盖 Transformer 的编码器‑解码器设计、多头注意力、掩码机制、位置编码，LLaMA 中的 RMSNorm、RoPE、KV 缓存、门控激活，同时包含 BPE、Word2Vec、余弦相似度等 NLP 基础。

## 📖 内容概览

- ✅ **Transformer 原理**：编码器与解码器结构、残差连接、层归一化 vs 批归一化、前馈神经网络
- ✅ **注意力机制**：自注意力、交叉注意力、多头注意力（MHA）、缩放点积、Softmax 数值稳定技巧
- ✅ **掩码机制**：填充掩码（Padding Mask）、因果掩码（Causal Mask）、自回归生成中的掩码扩展
- ✅ **位置编码**：正余弦位置编码（Sinusoidal）、公式推导与代码实现
- ✅ **训练与推理**：Teacher Forcing、自回归生成、温度参数、Top‑p 采样、KV 缓存优化
- ✅ **LLaMA 架构详解**：RMSNorm、旋转位置编码（RoPE）、MQA / GQA 注意力变体、SwiGLU 门控激活函数、无偏置设计
- ✅ **NLP 核心算法**：余弦相似度、Bag‑of‑Words、TF‑IDF、BPE / Unigram / SentencePiece 分词、Word2Vec / FastText / Doc2Vec
- ✅ **完整代码复现**：
  - **Encoder‑only**（文本分类器，基于 TransformerEncoder）
  - **Decoder‑only**（自回归文本生成，含因果掩码与 generate 方法）
  - **Encoder‑Decoder**（机器翻译，基于 nn.Transformer）
  - 使用 Huggingface 接口快速调用 BERT、GPT‑2、T5 等预训练模型
- ✅ **特殊 Token 详解**：CLS、SEP、MASK、PAD、BOS/EOS 等
- ✅ **手工构建词嵌入流程**：从文本清洗、子词分割到训练词向量的完整 pipeline

## 🚀 快速开始

```bash
# 直接阅读笔记（Markdown格式）
```

## 🔧 相关依赖库

```bash
pip install torch torchvision torchaudio transformers tokenizers sentencepiece tensorboard
```

## 🤝 贡献

欢迎提交 Issue 或 PR 以修正错误、补充新内容（如 MoE、RLHF、量化推理等）。  
如果本笔记对您有帮助，请点亮 ⭐ Star 支持一下～
