# day2小结

## 作业概述
本项目基于 `arxiv_mcp_server.py` 工具，专注收集 **TTS（文本转语音）和播客生成** 相关的学术论文。通过智能化的论文引用网络分析，从三个核心TTS概念出发，给定了每个概念的一些代表性论文，系统性地收集和整理了相关研究成果。

## 搜索策略

选择了三个精选的TTS核心概念作为搜索种子：

1. **文本转语音合成** (Text-to-Speech Synthesis)
   - 代表论文：VITS, FastSpeech, FastSpeech 2
   - 覆盖：端到端模型、声学模型、基础TTS技术

2. **多说话人TTS** (Multi-speaker TTS)
   - 代表论文：YourTTS, AdaSpeech, VALL-E
   - 覆盖：说话人适应、零样本合成、声音克隆

3. **情感语音合成** (Emotional Speech Synthesis)
   - 代表论文：EmotiVoice, StyleTTS, 可控情感TTS
   - 覆盖：情感控制、韵律调节、风格化合成