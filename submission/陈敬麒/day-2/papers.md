# TTS方向论文收集

基于种子论文及其引用网络自动收集，按方法类别分组：

## PPO相关论文

### VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling
- ArXiv链接: https://arxiv.org/abs/2408.15676
- 关键特点: Recent AIGC systems possess the capability to generate digital multimedia content based on human language instructions, such as text, image and video However, when it comes to speech, existing methods related to human instruction-to-speech generation exhibit two limitations
- 相关技术: PPO, Prompt

### ESCoT: Towards Interpretable Emotional Support Dialogue Systems
- ArXiv链接: https://arxiv.org/abs/2406.10960
- 关键特点: Understanding the reason for emotional support response is crucial for establishing connections between users and emotional support dialogue systems Previous works mostly focus on generating better responses but ignore interpretability, which is extremely important for constructing reliable dialogue systems
- 相关技术: PPO

### MM-TTS: Multi-modal Prompt based Style Transfer for Expressive Text-to-Speech Synthesis
- ArXiv链接: https://arxiv.org/abs/2312.10687
- 关键特点: The style transfer task in Text-to-Speech (TTS) refers to the process of transferring style information into text content to generate corresponding speech with a specific style However, most existing style transfer approaches are either based on fixed emotional labels or reference speech clips, which cannot achieve flexible style transfer
- 相关技术: PPO, Prompt

### PromptTTS 2: Describing and Generating Voices with Text Prompt
- ArXiv链接: https://arxiv.org/abs/2309.02285
- 关键特点: Speech conveys more information than text, as the same word can be uttered in various voices to convey diverse information Compared to traditional text-to-speech (TTS) methods relying on speech prompts (reference speech) for voice variability, using text prompts (descriptions) is more user-friendly since speech prompts can be hard to find or may not exist at all
- 相关技术: PPO, Prompt

### On the Opportunities and Risks of Foundation Models
- ArXiv链接: https://arxiv.org/abs/2108.07258
- 关键特点: AI is undergoing a paradigm shift with the rise of models (e g
- 相关技术: PPO, Search

## 非RL方法论文

### TextrolSpeech: A Text Style Control Speech Corpus with Codec Language Text-to-Speech Models
- ArXiv链接: https://arxiv.org/abs/2308.14430
- 关键特点: Recently, there has been a growing interest in the field of controllable Text-to-Speech (TTS) While previous studies have relied on users providing specific style factor values based on acoustic knowledge or selecting reference speeches that meet certain requirements, generating speech solely from natural text prompts has emerged as a new challenge for researchers
- 相关技术: Prompt, Search

### Imaginary Voice: Face-Styled Diffusion Model for Text-to-Speech
- ArXiv链接: https://arxiv.org/abs/2302.13700
- 关键特点: The goal of this work is zero-shot text-to-speech synthesis, with speaking styles and voices learnt from facial characteristics Inspired by the natural fact that people can imagine the voice of someone when they look at his or her face, we introduce a face-styled diffusion text-to-speech (TTS) model within a unified framework learnt from visible attributes, called Face-TTS
- 相关技术: Fine-tuning

### InstructTTS: Modelling Expressive TTS in Discrete Latent Space With Natural Language Style Prompt
- ArXiv链接: https://arxiv.org/abs/2301.13662
- 关键特点: Expressive text-to-speech (TTS) aims to synthesize speech with varying speaking styles to better reflect human speech patterns In this study, we attempt to use natural language as a style prompt to control the styles in the synthetic speech, e
- 相关技术: Prompt

### Prompt-to-Prompt Image Editing with Cross Attention Control
- ArXiv链接: https://arxiv.org/abs/2208.01626
- 关键特点: Recent large-scale text-driven synthesis models have attracted much attention thanks to their remarkable capabilities of generating highly diverse images that follow given text prompts Such text-based synthesis methods are particularly appealing to humans who are used to verbally describe their intent
- 相关技术: RL, Prompt

### Chain of Thought Prompting Elicits Reasoning in Large Language Models
- ArXiv链接: https://arxiv.org/abs/2201.11903
- 关键特点: We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting
- 相关技术: Prompt, Chain of Thought

## 综述类论文

### Audio Description Generation in the Era of LLMs and VLMs: A Review of Transferable Generative AI Technologies
- ArXiv链接: https://arxiv.org/abs/2410.08860
- 关键特点: Audio descriptions (ADs) function as acoustic commentaries designed to assist blind persons and persons with visual impairments in accessing digital media content on television and in movies, among other settings As an accessibility service typically provided by trained AD professionals, the generation of ADs demands significant human effort, making the process both time-consuming and costly
- 相关技术: RL, Search

### Video Summarization Using Deep Neural Networks: A Survey
- ArXiv链接: https://arxiv.org/abs/2101.06072
- 关键特点: Video summarization technologies aim to create a concise and complete synopsis by selecting the most informative parts of the video content Several approaches have been developed over the last couple of decades, and the current state of the art is represented by methods that rely on modern deep neural network architectures
- 相关技术: Survey, Search

## TTS端到端模型

### Tacotron: Towards End-to-End Speech Synthesis
- ArXiv链接: https://arxiv.org/abs/1703.10135v2
- 关键特点: A text-to-speech synthesis system typically consists of multiple stages, such as a text analysis frontend, an acoustic model and an audio synthesis module
- 相关技术: N/A

### StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models
- ArXiv链接: https://arxiv.org/abs/2306.07691
- 关键特点: In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models
- 相关技术: N/A

### YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone
- ArXiv链接: https://arxiv.org/abs/2112.02418
- 关键特点: YourTTS brings the power of a multilingual approach to the task of zero-shot multi-speaker TTS Our method builds upon the VITS model and adds several novel modifications for zero-shot multi-speaker and multilingual training
- 相关技术: N/A

### Multimodal Transformer for Unaligned Multimodal Language Sequences
- ArXiv链接: https://arxiv.org/abs/1906.00295
- 关键特点: Human language is often multimodal, which comprehends a mixture of natural language, facial gestures, and acoustic behaviors However, two major challenges in modeling such multimodal human language time-series data exist: 1) inherent data non-alignment due to variable sampling rates for the sequences from each modality; and 2) long-range dependencies between elements across modalities
- 相关技术: N/A

## 声码器

### Vocoder-Based Speech Synthesis from Silent Videos
- ArXiv链接: https://arxiv.org/abs/2004.02541
- 关键特点: Both acoustic and visual information influence human perception of speech For this reason, the lack of audio in a video sequence determines an extremely low speech intelligibility for untrained lip readers
- 相关技术: N/A

## 风格迁移

### V-CASS: Vision-context-aware Expressive Speech Synthesis for Enhancing User Understanding of Videos
- ArXiv链接: https://arxiv.org/abs/2506.16716
- 关键特点: Automatic video commentary systems are widely used on multimedia social media platforms to extract factual information about video content However, current systems may overlook essential para-linguistic cues, including emotion and attitude, which are critical for fully conveying the meaning of visual content
- 相关技术: RL

### V-CASS: Vision-context-aware Expressive Speech Synthesis for Enhancing
  User Understanding of Videos
- ArXiv链接: https://arxiv.org/abs/2506.16716v1
- 关键特点: Automatic video commentary systems are widely used on multimedia social media platforms to extract factual information about video content
- 相关技术: RL

### EALD-MLLM: Emotion Analysis in Long-sequential and De-identity videos with Multi-modal Large Language Model
- ArXiv链接: https://arxiv.org/abs/2405.00574
- 关键特点: Emotion AI is the ability of computers to understand human emotional states Existing works have achieved promising progress, but two limitations remain to be solved: 1) Previous studies have been more focused on short sequential video emotion analysis while overlooking long sequential video
- 相关技术: RL, Search

### ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision
- ArXiv链接: https://arxiv.org/abs/2102.03334
- 关键特点: Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision (e
- 相关技术: N/A

## 其他

### MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation
- ArXiv链接: https://arxiv.org/abs/2212.09478
- 关键特点: We propose the first joint audio-video generation framework that brings engaging watching and listening experiences simultaneously, towards high-quality realistic videos To generate joint audio-video pairs, we propose a novel Multi-Modal Diffusion model (i
- 相关技术: Search

### Learning Transferable Visual Models From Natural Language Supervision
- ArXiv链接: https://arxiv.org/abs/2103.00020
- 关键特点: State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept
- 相关技术: N/A

### AutoFoley: Artificial Synthesis of Synchronized Sound Tracks for Silent Videos With Deep Learning
- ArXiv链接: https://arxiv.org/abs/2002.10981
- 关键特点: In movie productions, the Foley artist is responsible for creating an overlay soundtrack that helps the movie come alive for the audience This requires the artist identify sounds that enhance the experience for the listener, reinforcing the director's intention for the scene
- 相关技术: RL, Search

### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- ArXiv链接: https://arxiv.org/abs/1908.10084
- 关键特点: BERT (Devlin et al , 2018) and RoBERTa (Liu et al
- 相关技术: Search

---

**统计总结:**
- PPO相关论文: 5篇论文
- 非RL方法论文: 5篇论文
- 综述类论文: 2篇论文
- TTS端到端模型: 4篇论文
- 声码器: 1篇论文
- 风格迁移: 4篇论文
- 其他: 4篇论文

**主要趋势:**
1. 端到端TTS模型（如Tacotron、FastSpeech、VITS）持续推动语音合成自然度和效率提升
2. 声码器（Vocoder）技术进步带来高保真语音生成
3. 多说话人、跨语言、风格迁移等多样化TTS需求成为研究热点
4. Diffusion、Prompt等新范式和大模型方法在TTS中快速发展
5. 轻量化、实时推理和模型压缩技术满足实际应用需求
