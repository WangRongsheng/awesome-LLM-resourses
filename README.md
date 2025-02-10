![](./assets/logo2.png)

<p align="center">全世界最好的大语言模型资源汇总 持续更新</p>

<p align="center">
<a href="https://info.flagcounter.com/U6Yn"><img src="https://s11.flagcounter.com/count2/U6Yn/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_1/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
</p>

<details>
<summary>Check More Information</summary>
<p align="center">
  <a href="https://github.com/WangRongsheng/awesome-LLM-resourses"><img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg ></a>
  <a href="https://github.com/WangRongsheng/awesome-LLM-resourses"><img src=https://img.shields.io/github/forks/WangRongsheng/awesome-LLM-resourses.svg?style=social ></a>
  <a href="https://github.com/WangRongsheng/awesome-LLM-resourses"><img src=https://img.shields.io/github/stars/WangRongsheng/awesome-LLM-resourses.svg?style=social ></a>
  <a href="https://github.com/WangRongsheng/awesome-LLM-resourses"><img src=https://img.shields.io/github/watchers/WangRongsheng/awesome-LLM-resourses.svg?style=social ></a>
</p>

</details>

<p align="center">
  <a href="https://www.wangrs.site/awesome-LLM-resourses/">[在线阅读]</a>
</p>

![](https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67)

#### Contents

- [数据 Data](#数据-Data)
- [微调 Fine-Tuning](#微调-Fine-Tuning)
- [推理 Inference](#推理-Inference)
- [评估 Evaluation](#评估-Evaluation)
- [体验 Usage](#体验-Usage)
- [知识库 RAG](#知识库-RAG)
- [智能体 Agents](#智能体-Agents)
- [搜索 Search](#搜索-Search)
- [书籍 Book](#书籍-Book)
- [课程 Course](#课程-Course)
- [教程 Tutorial](#教程-Tutorial)
- [论文 Paper](#论文-Paper)
- [社区 Community](#社区-Community)
- [Open o1](#Open-o1)
- [Small Language Model](#Small-Language-Model)
- [Small Vision Language Model](#Small-Vision-Language-Model)
- [Tips](#tips)

![](https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67)

## 数据 Data

> [!NOTE]
> 
> 此处命名为`数据`，但这里并没有提供具体数据集，而是提供了处理获取大规模数据的方法


1. [AotoLabel](https://github.com/refuel-ai/autolabel): Label, clean and enrich text datasets with LLMs.
2. [LabelLLM](https://github.com/opendatalab/LabelLLM): The Open-Source Data Annotation Platform.
3. [data-juicer](https://github.com/modelscope/data-juicer): A one-stop data processing system to make data higher-quality, juicier, and more digestible for LLMs!
4. [OmniParser](https://github.com/jf-tech/omniparser): a native Golang ETL streaming parser and transform library for CSV, JSON, XML, EDI, text, etc.
5. [MinerU](https://github.com/opendatalab/MinerU): MinerU is a one-stop, open-source, high-quality data extraction tool, supports PDF/webpage/e-book extraction.
6. [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit): A Comprehensive Toolkit for High-Quality PDF Content Extraction.
7. [Parsera](https://github.com/raznem/parsera): Lightweight library for scraping web-sites with LLMs.
8. [Sparrow](https://github.com/katanaml/sparrow): Sparrow is an innovative open-source solution for efficient data extraction and processing from various documents and images.
9. [Docling](https://github.com/DS4SD/docling): Get your documents ready for gen AI.
10. [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0): OCR Model.
11. [LLM Decontaminator](https://github.com/lm-sys/llm-decontaminator): Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.
12. [DataTrove](https://github.com/huggingface/datatrove): DataTrove is a library to process, filter and deduplicate text data at a very large scale.
13. [llm-swarm](https://github.com/huggingface/llm-swarm/tree/main/examples/textbooks): Generate large synthetic datasets like [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia).
14. [Distilabel](https://github.com/argilla-io/distilabel): Distilabel is a framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers.
15. [Common-Crawl-Pipeline-Creator](https://huggingface.co/spaces/lhoestq/Common-Crawl-Pipeline-Creator): The Common Crawl Pipeline Creator.
16. [Tabled](https://github.com/VikParuchuri/tabled): Detect and extract tables to markdown and csv.
17. [Zerox](https://github.com/getomni-ai/zerox): Zero shot pdf OCR with gpt-4o-mini.
18. [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO): Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception.
19. [TensorZero](https://github.com/tensorzero/tensorzero): make LLMs improve through experience.
20. [Promptwright](https://github.com/StacklokLabs/promptwright): Generate large synthetic data using a local LLM.
21. [pdf-extract-api](https://github.com/CatchTheTornado/pdf-extract-api): Document (PDF) extraction and parse API using state of the art modern OCRs + Ollama supported models.
22. [pdf2htmlEX](https://github.com/pdf2htmlEX/pdf2htmlEX): Convert PDF to HTML without losing text or format.
23. [Extractous](https://github.com/yobix-ai/extractous): Fast and efficient unstructured data extraction. Written in Rust with bindings for many languages.
24. [MegaParse](https://github.com/QuivrHQ/MegaParse): File Parser optimised for LLM Ingestion with no loss.
25. [MarkItDown](https://github.com/microsoft/markitdown): Python tool for converting files and office documents to Markdown.
26. [datasketch](https://github.com/ekzhu/datasketch): datasketch gives you probabilistic data structures that can process and search very large amount of data super fast, with little loss of accuracy.
27. [semhash](https://github.com/MinishLab/semhash): lightweight and flexible tool for deduplicating datasets using semantic similarity.
28. [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2): a 1.5B parameter language model that converts raw HTML into beautifully formatted markdown or JSON.
29. [Bespoke Curator](https://github.com/bespokelabsai/curator): Data Curation for Post-Training & Structured Data Extraction.
30. [LangKit](https://github.com/whylabs/langkit): An open-source toolkit for monitoring Large Language Models (LLMs). Extracts signals from prompts & responses, ensuring safety & security.
31. [Curator](https://github.com/bespokelabsai/curator): Synthetic Data curation for post-training and structured data extraction.

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 微调 Fine-Tuning

1. [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Unify Efficient Fine-Tuning of 100+ LLMs.
2. [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory): Unify Efficient Fine-Tuning of 100+ LLMs. (add Sequence Parallelism for supporting long context training)
4. [unsloth](https://github.com/unslothai/unsloth): 2-5X faster 80% less memory LLM finetuning.
5. [TRL](https://huggingface.co/docs/trl/index): Transformer Reinforcement Learning.
6. [Firefly](https://github.com/yangjianxin1/Firefly): Firefly: 大模型训练工具，支持训练数十种大模型
7. [Xtuner](https://github.com/InternLM/xtuner): An efficient, flexible and full-featured toolkit for fine-tuning large models.
8. [torchtune](https://github.com/pytorch/torchtune): A Native-PyTorch Library for LLM Fine-tuning.
9. [Swift](https://github.com/modelscope/swift): Use PEFT or Full-parameter to finetune 200+ LLMs or 15+ MLLMs.
10. [AutoTrain](https://huggingface.co/autotrain): A new way to automatically train, evaluate and deploy state-of-the-art Machine Learning models.
11. [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework (Support 70B+ full tuning & LoRA & Mixtral & KTO).
12. [Ludwig](https://github.com/ludwig-ai/ludwig): Low-code framework for building custom LLMs, neural networks, and other AI models.
13. [mistral-finetune](https://github.com/mistralai/mistral-finetune): A light-weight codebase that enables memory-efficient and performant finetuning of Mistral's models.
14. [aikit](https://github.com/sozercan/aikit): Fine-tune, build, and deploy open-source LLMs easily!
15. [H2O-LLMStudio](https://github.com/h2oai/h2o-llmstudio): H2O LLM Studio - a framework and no-code GUI for fine-tuning LLMs.
16. [LitGPT](https://github.com/Lightning-AI/litgpt): Pretrain, finetune, deploy 20+ LLMs on your own data. Uses state-of-the-art techniques: flash attention, FSDP, 4-bit, LoRA, and more.
17. [LLMBox](https://github.com/RUCAIBox/LLMBox): A comprehensive library for implementing LLMs, including a unified training pipeline and comprehensive model evaluation.
18. [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP): Easy-to-use and powerful NLP and LLM library.
19. [workbench-llamafactory](https://github.com/NVIDIA/workbench-llamafactory): This is an NVIDIA AI Workbench example project that demonstrates an end-to-end model development workflow using Llamafactory.
20. [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework (70B+ PPO Full Tuning & Iterative DPO & LoRA & Mixtral).
21. [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory): A Framework of Small-scale Large Multimodal Models.
22. [LLM-Foundry](https://github.com/mosaicml/llm-foundry): LLM training code for Databricks foundation models.
23. [lmms-finetune](https://github.com/zjysteven/lmms-finetune): A unified codebase for finetuning (full, lora) large multimodal models, supporting llava-1.5, qwen-vl, llava-interleave, llava-next-video, phi3-v etc.
24. [Simplifine](https://github.com/simplifine-llm/Simplifine): Simplifine lets you invoke LLM finetuning with just one line of code using any Hugging Face dataset or model.
25. [Transformer Lab](https://github.com/transformerlab/transformerlab-app): Open Source Application for Advanced LLM Engineering: interact, train, fine-tune, and evaluate large language models on your own computer.
26. [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Efficient Triton Kernels for LLM Training.
27. [ChatLearn](https://github.com/alibaba/ChatLearn): A flexible and efficient training framework for large-scale alignment.
28. [nanotron](https://github.com/huggingface/nanotron): Minimalistic large language model 3D-parallelism training.
29. [Proxy Tuning](https://github.com/alisawuffles/proxy-tuning): Tuning Language Models by Proxy.
30. [Effective LLM Alignment](https://github.com/VikhrModels/effective_llm_alignment/): Effective LLM Alignment Toolkit.
31. [Autotrain-advanced](https://github.com/huggingface/autotrain-advanced)
32. [Meta Lingua](https://github.com/facebookresearch/lingua): a lean, efficient, and easy-to-hack codebase to research LLMs.
33. [Vision-LLM Alignemnt](https://github.com/NiuTrans/Vision-LLM-Alignment): This repository contains the code for SFT, RLHF, and DPO, designed for vision-based LLMs, including the LLaVA models and the LLaMA-3.2-vision models.
34. [finetune-Qwen2-VL](https://github.com/zhangfaen/finetune-Qwen2-VL): Quick Start for Fine-tuning or continue pre-train Qwen2-VL Model.
35. [Online-RLHF](https://github.com/RLHFlow/Online-RLHF): A recipe for online RLHF and online iterative DPO.
36. [InternEvo](https://github.com/InternLM/InternEvo): an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies.
37. [veRL](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning for LLM.
38. [Axolotl](https://axolotl-ai-cloud.github.io/axolotl/): Axolotl is designed to work with YAML config files that contain everything you need to preprocess a dataset, train or fine-tune a model, run model inference or evaluation, and much more.
39. [Oumi](https://github.com/oumi-ai/oumi): Everything you need to build state-of-the-art foundation models, end-to-end.
40. [Kiln](https://github.com/Kiln-AI/Kiln): The easiest tool for fine-tuning LLM models, synthetic data generation, and collaborating on datasets.

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 推理 Inference

1. [ollama](https://github.com/ollama/ollama): Get up and running with Llama 3, Mistral, Gemma, and other large language models.
2. [Open WebUI](https://github.com/open-webui/open-webui): User-friendly WebUI for LLMs (Formerly Ollama WebUI).
3. [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.
4. [Xinference](https://github.com/xorbitsai/inference): A powerful and versatile library designed to serve language, speech recognition, and multimodal models.
5. [LangChain](https://github.com/langchain-ai/langchain): Build context-aware reasoning applications.
6. [LlamaIndex](https://github.com/run-llama/llama_index): A data framework for your LLM applications.
7. [lobe-chat](https://github.com/lobehub/lobe-chat): an open-source, modern-design LLMs/AI chat framework. Supports Multi AI Providers, Multi-Modals (Vision/TTS) and plugin system.
8. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.
9. [vllm](https://github.com/vllm-project/vllm): A high-throughput and memory-efficient inference and serving engine for LLMs.
10. [LlamaChat](https://github.com/alexrozanski/LlamaChat): Chat with your favourite LLaMA models in a native macOS app.
11. [NVIDIA ChatRTX](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/): ChatRTX is a demo app that lets you personalize a GPT large language model (LLM) connected to your own content—docs, notes, or other data.
12. [LM Studio](https://lmstudio.ai/): Discover, download, and run local LLMs.
13. [chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx): Chat with your data natively on Apple Silicon using MLX Framework.
14. [LLM Pricing](https://llmpricecheck.com/): Quickly Find the Perfect Large Language Models (LLM) API for Your Budget! Use Our Free Tool for Instant Access to the Latest Prices from Top Providers.
15. [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter): A natural language interface for computers.
16. [Chat-ollama](https://github.com/sugarforever/chat-ollama): An open source chatbot based on LLMs. It supports a wide range of language models, and knowledge base management.
17. [chat-ui](https://github.com/huggingface/chat-ui): Open source codebase powering the HuggingChat app.
18. [MemGPT](https://github.com/cpacker/MemGPT): Create LLM agents with long-term memory and custom tools.
19. [koboldcpp](https://github.com/LostRuins/koboldcpp): A simple one-file way to run various GGML and GGUF models with KoboldAI's UI.
20. [LLMFarm](https://github.com/guinmoon/LLMFarm): llama and other large language models on iOS and MacOS offline using GGML library.
21. [enchanted](https://github.com/AugustDev/enchanted): Enchanted is iOS and macOS app for chatting with private self hosted language models such as Llama2, Mistral or Vicuna using Ollama.
22. [Flowise](https://github.com/FlowiseAI/Flowise): Drag & drop UI to build your customized LLM flow.
23. [Jan](https://github.com/janhq/jan): Jan is an open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM).
24. [LMDeploy](https://github.com/InternLM/lmdeploy): LMDeploy is a toolkit for compressing, deploying, and serving LLMs.
25. [RouteLLM](https://github.com/lm-sys/RouteLLM): A framework for serving and evaluating LLM routers - save LLM costs without compromising quality!
26. [MInference](https://github.com/microsoft/MInference): About
To speed up Long-context LLMs' inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accuracy.
27. [Mem0](https://github.com/mem0ai/mem0): The memory layer for Personalized AI.
28. [SGLang](https://github.com/sgl-project/sglang): SGLang is yet another fast serving framework for large language models and vision language models.
29. [AirLLM](https://github.com/lyogavin/airllm): AirLLM optimizes inference memory usage, allowing 70B large language models to run inference on a single 4GB GPU card without quantization, distillation and pruning. And you can run 405B Llama3.1 on 8GB vram now.
30. [LLMHub](https://github.com/jmather/llmhub): LLMHub is a lightweight management platform designed to streamline the operation and interaction with various language models (LLMs).
31. [YuanChat](https://github.com/IEIT-Yuan/YuanChat)
32. [LiteLLM](https://github.com/BerriAI/litellm): Call all LLM APIs using the OpenAI format [Bedrock, Huggingface, VertexAI, TogetherAI, Azure, OpenAI, Groq etc.]
33. [GuideLLM](https://github.com/neuralmagic/guidellm): GuideLLM is a powerful tool for evaluating and optimizing the deployment of large language models (LLMs).
34. [LLM-Engines](https://github.com/jdf-prog/LLM-Engines): A unified inference engine for large language models (LLMs) including open-source models (VLLM, SGLang, Together) and commercial models (OpenAI, Mistral, Claude).
35. [OARC](https://github.com/Leoleojames1/ollama_agent_roll_cage): ollama_agent_roll_cage (OARC) is a local python agent fusing ollama llm's with Coqui-TTS speech models, Keras classifiers, Llava vision, Whisper recognition, and more to create a unified chatbot agent for local, custom automation.
36. [g1](https://github.com/bklieger-groq/g1): Using Llama-3.1 70b on Groq to create o1-like reasoning chains.
37. [MemoryScope](https://github.com/modelscope/MemoryScope): MemoryScope provides LLM chatbots with powerful and flexible long-term memory capabilities, offering a framework for building such abilities.
38. [OpenLLM](https://github.com/bentoml/OpenLLM): Run any open-source LLMs, such as Llama 3.1, Gemma, as OpenAI compatible API endpoint in the cloud.
39. [Infinity](https://github.com/infiniflow/infinity): The AI-native database built for LLM applications, providing incredibly fast hybrid search of dense embedding, sparse embedding, tensor and full-text.
40. [optillm](https://github.com/codelion/optillm): an OpenAI API compatible optimizing inference proxy which implements several state-of-the-art techniques that can improve the accuracy and performance of LLMs.
41. [LLaMA Box](https://github.com/gpustack/llama-box): LLM inference server implementation based on llama.cpp.
42. [ZhiLight](https://github.com/zhihu/ZhiLight): A highly optimized inference acceleration engine for Llama and its variants.
43. [DashInfer](https://github.com/modelscope/dash-infer): DashInfer is a native LLM inference engine aiming to deliver industry-leading performance atop various hardware architectures.
44. [LocalAI](https://github.com/mudler/LocalAI): The free, Open Source alternative to OpenAI, Claude and others. Self-hosted and local-first. Drop-in replacement for OpenAI, running on consumer-grade hardware. No GPU required.


<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 评估 Evaluation

1. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models.
2. [opencompass](https://github.com/open-compass/opencompass): OpenCompass is an LLM evaluation platform, supporting a wide range of models (Llama3, Mistral, InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.
3. [llm-comparator](https://github.com/PAIR-code/llm-comparator): LLM Comparator is an interactive data visualization tool for evaluating and analyzing LLM responses side-by-side, developed.
4. [EvalScope](https://github.com/modelscope/evalscope)
5. [Weave](https://weave-docs.wandb.ai/guides/core-types/evaluations): A lightweight toolkit for tracking and evaluating LLM applications.
6. [MixEval](https://github.com/Psycoy/MixEval/): Deriving Wisdom of the Crowd from LLM Benchmark Mixtures.
7. [Evaluation guidebook](https://github.com/huggingface/evaluation-guidebook): If you've ever wondered how to make sure an LLM performs well on your specific task, this guide is for you!
8. [Ollama Benchmark](https://github.com/aidatatools/ollama-benchmark): LLM Benchmark for Throughput via Ollama (Local LLMs).
9. [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): Open-source evaluation toolkit of large vision-language models (LVLMs), support ~100 VLMs, 40+ benchmarks.
10. [AGI-Eval](https://agi-eval.cn/mvp/home)
11. [EvalScope](https://github.com/modelscope/evalscope): A streamlined and customizable framework for efficient large model evaluation and performance benchmarking.

`LLM API 服务平台`：
1. [Groq](https://groq.com/)
2. [硅基流动](https://cloud.siliconflow.cn/models)
3. [火山引擎](https://www.volcengine.com/product/ark)
4. [文心千帆](https://qianfan.cloud.baidu.com/)
5. [DashScope](https://dashscope.aliyun.com/)
6. [aisuite](https://github.com/andrewyng/aisuite)
7. [DeerAPI](https://www.deerapi.com/)
8. [Qwen-Chat](https://chat.qwenlm.ai/)
9. [DeepSeek-v3](https://www.deepseek.com/)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 体验 Usage

1. [LMSYS Chatbot Arena: Benchmarking LLMs in the Wild](https://arena.lmsys.org/)
2. [CompassArena 司南大模型竞技场](https://modelscope.cn/studios/opencompass/CompassArena/summary)
3. [琅琊榜](https://langyb.com/)
4. [Huggingface Spaces](https://huggingface.co/spaces)
5. [WiseModel Spaces](https://wisemodel.cn/spaces)
6. [Poe](https://poe.com/)
7. [林哥的大模型野榜](https://lyihub.com/)
8. [OpenRouter](https://openrouter.ai/)
9. [AnyChat](https://huggingface.co/spaces/akhaliq/anychat)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 知识库 RAG

1. [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm): The all-in-one AI app for any LLM with full RAG and AI Agent capabilites.
2. [MaxKB](https://github.com/1Panel-dev/MaxKB): 基于 LLM 大语言模型的知识库问答系统。开箱即用，支持快速嵌入到第三方业务系统
3. [RAGFlow](https://github.com/infiniflow/ragflow): An open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding.
4. [Dify](https://github.com/langgenius/dify): An open-source LLM app development platform. Dify's intuitive interface combines AI workflow, RAG pipeline, agent capabilities, model management, observability features and more, letting you quickly go from prototype to production.
5. [FastGPT](https://github.com/labring/FastGPT): A knowledge-based platform built on the LLM, offers out-of-the-box data processing and model invocation capabilities, allows for workflow orchestration through Flow visualization.
6. [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat): 基于 Langchain 与 ChatGLM 等不同大语言模型的本地知识库问答
7. [QAnything](https://github.com/netease-youdao/QAnything): Question and Answer based on Anything.
8. [Quivr](https://github.com/QuivrHQ/quivr): A personal productivity assistant (RAG) ⚡️🤖 Chat with your docs (PDF, CSV, ...) & apps using Langchain, GPT 3.5 / 4 turbo, Private, Anthropic, VertexAI, Ollama, LLMs, Groq that you can share with users ! Local & Private alternative to OpenAI GPTs & ChatGPT powered by retrieval-augmented generation.
9. [RAG-GPT](https://github.com/open-kf/rag-gpt): RAG-GPT, leveraging LLM and RAG technology, learns from user-customized knowledge bases to provide contextually relevant answers for a wide range of queries, ensuring rapid and accurate information retrieval.
10. [Verba](https://github.com/weaviate/Verba): Retrieval Augmented Generation (RAG) chatbot powered by Weaviate.
11. [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG): A Python Toolkit for Efficient RAG Research.
12. [GraphRAG](https://github.com/microsoft/graphrag): A modular graph-based Retrieval-Augmented Generation (RAG) system.
13. [LightRAG](https://github.com/SylphAI-Inc/LightRAG): LightRAG helps developers with both building and optimizing Retriever-Agent-Generator pipelines.
14. [GraphRAG-Ollama-UI](https://github.com/severian42/GraphRAG-Ollama-UI): GraphRAG using Ollama with Gradio UI and Extra Features.
15. [nano-GraphRAG](https://github.com/gusye1234/nano-graphrag): A simple, easy-to-hack GraphRAG implementation.
16. [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques): This repository showcases various advanced techniques for Retrieval-Augmented Generation (RAG) systems. RAG systems combine information retrieval with generative models to provide accurate and contextually rich responses.
17. [ragas](https://github.com/explodinggradients/ragas): Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines.
18. [kotaemon](https://github.com/Cinnamon/kotaemon): An open-source clean & customizable RAG UI for chatting with your documents. Built with both end users and developers in mind.
19. [RAGapp](https://github.com/ragapp/ragapp): The easiest way to use Agentic RAG in any enterprise.
20. [TurboRAG](https://github.com/MooreThreads/TurboRAG): Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text.
21. [LightRAG](https://github.com/HKUDS/LightRAG): Simple and Fast Retrieval-Augmented Generation.
22. [TEN](https://github.com/TEN-framework/ten_framework): the Next-Gen AI-Agent Framework, the world's first truly real-time multimodal AI agent framework.
23. [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG): RAG AutoML tool for automatically finding an optimal RAG pipeline for your data.
24. [KAG](https://github.com/OpenSPG/KAG): KAG is a knowledge-enhanced generation framework based on OpenSPG engine, which is used to build knowledge-enhanced rigorous decision-making and information retrieval knowledge services.
25. [Fast-GraphRAG](https://github.com/circlemind-ai/fast-graphrag): RAG that intelligently adapts to your use case, data, and queries.
26. [Tiny-GraphRAG](https://github.com/limafang/tiny-graphrag)
27. [DB-GPT GraphRAG](https://github.com/eosphoros-ai/DB-GPT/tree/main/dbgpt/storage/knowledge_graph): DB-GPT GraphRAG integrates both triplet-based knowledge graphs and document structure graphs while leveraging community and document retrieval mechanisms to enhance RAG capabilities, achieving comparable performance while consuming only 50% of the tokens required by Microsoft's GraphRAG. Refer to the DB-GPT [Graph RAG User Manual](http://docs.dbgpt.cn/docs/cookbook/rag/graph_rag_app_develop/) for details.
28. [Chonkie](https://github.com/bhavnicksm/chonkie): The no-nonsense RAG chunking library that's lightweight, lightning-fast, and ready to CHONK your texts.
29. [RAGLite](https://github.com/superlinear-ai/raglite): RAGLite is a Python toolkit for Retrieval-Augmented Generation (RAG) with PostgreSQL or SQLite.
30. [KAG](https://github.com/OpenSPG/KAG): KAG is a logical form-guided reasoning and retrieval framework based on OpenSPG engine and LLMs.
31. [CAG](https://github.com/hhhuang/CAG): CAG leverages the extended context windows of modern large language models (LLMs) by preloading all relevant resources into the model’s context and caching its runtime parameters.
32. [MiniRAG](https://github.com/HKUDS/MiniRAG): an extremely simple retrieval-augmented generation framework that enables small models to achieve good RAG performance through heterogeneous graph indexing and lightweight topology-enhanced retrieval.
33. [XRAG](https://github.com/DocAILab/XRAG): a benchmarking framework designed to evaluate the foundational components of advanced Retrieval-Augmented Generation (RAG) systems.

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 智能体 Agents

1. [AutoGen](https://github.com/microsoft/autogen): AutoGen is a framework that enables the development of LLM applications using multiple agents that can converse with each other to solve tasks. [AutoGen AIStudio](https://autogen-studio.com/)
2. [CrewAI](https://github.com/joaomdmoura/crewAI): Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.
3. [Coze](https://www.coze.com/)
4. [AgentGPT](https://github.com/reworkd/AgentGPT): Assemble, configure, and deploy autonomous AI Agents in your browser.
5. [XAgent](https://github.com/OpenBMB/XAgent): An Autonomous LLM Agent for Complex Task Solving.
6. [MobileAgent](https://github.com/X-PLUG/MobileAgent): The Powerful Mobile Device Operation Assistant Family.
7. [Lagent](https://github.com/InternLM/lagent): A lightweight framework for building LLM-based agents.
8. [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent): Agent framework and applications built upon Qwen2, featuring Function Calling, Code Interpreter, RAG, and Chrome extension.
9. [LinkAI](https://link-ai.tech/portal): 一站式 AI 智能体搭建平台
10. [Baidu APPBuilder](https://appbuilder.cloud.baidu.com/)
11. [agentUniverse](https://github.com/alipay/agentUniverse): agentUniverse is a LLM multi-agent framework that allows developers to easily build multi-agent applications. Furthermore, through the community, they can exchange and share practices of patterns across different domains.
12. [LazyLLM](https://github.com/LazyAGI/LazyLLM): 低代码构建多Agent大模型应用的开发工具
13. [AgentScope](https://github.com/modelscope/agentscope): Start building LLM-empowered multi-agent applications in an easier way.
14. [MoA](https://github.com/togethercomputer/MoA): Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results.
15. [Agently](https://github.com/Maplemx/Agently): AI Agent Application Development Framework.
16. [OmAgent](https://github.com/om-ai-lab/OmAgent): A multimodal agent framework for solving complex tasks.
17. [Tribe](https://github.com/StreetLamb/tribe): No code tool to rapidly build and coordinate multi-agent teams.
18. [CAMEL](https://github.com/camel-ai/camel): First LLM multi-agent framework and an open-source community dedicated to finding the scaling law of agents.
19. [PraisonAI](https://github.com/MervinPraison/PraisonAI/): PraisonAI application combines AutoGen and CrewAI or similar frameworks into a low-code solution for building and managing multi-agent LLM systems, focusing on simplicity, customisation, and efficient human-agent collaboration.
20. [IoA](https://github.com/openbmb/ioa): An open-source framework for collaborative AI agents, enabling diverse, distributed agents to team up and tackle complex tasks through internet-like connectivity.
21. [llama-agentic-system ](https://github.com/meta-llama/llama-agentic-system): Agentic components of the Llama Stack APIs.
22. [Agent Zero](https://github.com/frdel/agent-zero): Agent Zero is not a predefined agentic framework. It is designed to be dynamic, organically growing, and learning as you use it.
23. [Agents](https://github.com/aiwaves-cn/agents): An Open-source Framework for Data-centric, Self-evolving Autonomous Language Agents.
24. [AgentScope](https://github.com/modelscope/agentscope): Start building LLM-empowered multi-agent applications in an easier way.
25. [FastAgency](https://github.com/airtai/fastagency): The fastest way to bring multi-agent workflows to production.
26. [Swarm](https://github.com/openai/swarm): Framework for building, orchestrating and deploying multi-agent systems. Managed by OpenAI Solutions team. Experimental framework.
27. [Agent-S](https://github.com/simular-ai/Agent-S): an open agentic framework that uses computers like a human.
28. [PydanticAI](https://github.com/pydantic/pydantic-ai): Agent Framework / shim to use Pydantic with LLMs.
29. [Agentarium](https://github.com/Thytu/Agentarium): open-source framework for creating and managing simulations populated with AI-powered agents.
30. [smolagents](https://github.com/huggingface/smolagents): a barebones library for agents. Agents write python code to call tools and orchestrate other agents.

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 搜索 Search

1. [OpenSearch GPT](https://github.com/supermemoryai/opensearch-ai): SearchGPT / Perplexity clone, but personalised for you.
2. [MindSearch](https://github.com/InternLM/MindSearch): An LLM-based Multi-agent Framework of Web Search Engine (like Perplexity.ai Pro and SearchGPT).
3. [nanoPerplexityAI](https://github.com/Yusuke710/nanoPerplexityAI): The simplest open-source implementation of perplexity.ai.
4. [curiosity](https://github.com/jank/curiosity): Try to build a Perplexity-like user experience.
5. [MiniPerplx](https://github.com/zaidmukaddam/miniperplx): A minimalistic AI-powered search engine that helps you find information on the internet.

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 书籍 Book

1. [《大规模语言模型：从理论到实践》](https://intro-llm.github.io/)
2. [《大语言模型》](https://llmbook-zh.github.io/)
3. [《动手学大模型Dive into LLMs》](https://github.com/Lordog/dive-into-llms)
4. [《动手做AI Agent》](https://book.douban.com/subject/36884058/)
5. [《Build a Large Language Model (From Scratch)》](https://github.com/rasbt/LLMs-from-scratch)
6. [《多模态大模型》](https://github.com/HCPLab-SYSU/Book-of-MLM)
7. [《Generative AI Handbook: A Roadmap for Learning Resources》](https://genai-handbook.github.io/)
8. [《Understanding Deep Learning》](https://udlbook.github.io/udlbook/)
9. [《Illustrated book to learn about Transformers & LLMs》](https://www.reddit.com/r/MachineLearning/comments/1ew1hws/p_illustrated_book_to_learn_about_transformers/)
10. [《Building LLMs for Production: Enhancing LLM Abilities and Reliability with Prompting, Fine-Tuning, and RAG》](https://www.amazon.com/Building-LLMs-Production-Reliability-Fine-Tuning/dp/B0D4FFPFW8?crid=7OAXELUKGJE4&dib=eyJ2IjoiMSJ9.Qr3e3VSH8LSo_j1M7sV7GfS01q_W1LDYd2uGlvGJ8CW-t4DTlng6bSeOlZBryhp6HJN5K1HqWMVVgabU2wz2i9yLpy_AuaZN-raAEbenKx2NHtzZA3A4k-N7GpnldF1baCarA_V1CRF-aCdc9_3WSX7SaEzmpyDv22TTyltcKT74HAb2KiQqBGLhQS3cEAnzChcqGa1Xp-XhbMnplVwT7xZLApE3tGLhDOgi5GmSi9w.8SY_4NBEkm68YF4GwhDnz0r81ZB1d8jr-gK9IMJE5AE&dib_tag=se&keywords=building+llms+for+production&qid=1716376414&sprefix=building+llms+for+production,aps,101&sr=8-1&linkCode=sl1&tag=whatsai06-20&linkId=ee102fda07a0eb51710fcdd8b8d20c28&language=en_US&ref_=as_li_ss_tl)
11. [《大型语言模型实战指南：应用实践与场景落地》](https://github.com/liucongg/LLMsBook)
12. [《Hands-On Large Language Models》](https://github.com/handsOnLLM/Hands-On-Large-Language-Models)
13. [《自然语言处理：大模型理论与实践》](https://nlp-book.swufenlp.group/)
14. [《动手学强化学习》](https://hrl.boyuai.com/)
15. [《面向开发者的LLM入门教程》](https://datawhalechina.github.io/llm-cookbook/#/)
16. [《大模型基础》](https://github.com/ZJU-LLMs/Foundations-of-LLMs)
17. [Taming LLMs: A Practical Guide to LLM Pitfalls with Open Source Software ](https://www.tamingllms.com/)
18. [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 课程 Course

> [LLM Resources Hub](https://llmresourceshub.vercel.app/)

1. [斯坦福 CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
2. [吴恩达: Generative AI for Everyone](https://www.deeplearning.ai/courses/generative-ai-for-everyone/)
3. [吴恩达: LLM series of courses](https://learn.deeplearning.ai/)
4. [ACL 2023 Tutorial: Retrieval-based Language Models and Applications](https://acl2023-retrieval-lm.github.io/)
5. [llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.](https://github.com/mlabonne/llm-course)
6. [微软: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners)
7. [微软: State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)
8. [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)
9. [清华 NLP 刘知远团队大模型公开课](https://www.bilibili.com/video/BV1UG411p7zv/?vd_source=c739db1ebdd361d47af5a0b8497417db)
10. [斯坦福 CS25: Transformers United V4](https://web.stanford.edu/class/cs25/)
11. [斯坦福 CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/)
12. [普林斯顿 COS 597G (Fall 2022): Understanding Large Language Models](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
13. [约翰霍普金斯 CS 601.471/671 NLP: Self-supervised Models](https://self-supervised.cs.jhu.edu/sp2023/index.html)
14. [李宏毅 GenAI课程](https://www.youtube.com/watch?v=yiY4nPOzJEg&list=PLJV_el3uVTsOePyfmkfivYZ7Rqr2nMk3W)
15. [openai-cookbook](https://github.com/openai/openai-cookbook): Examples and guides for using the OpenAI API.
16. [Hands on llms](https://github.com/iusztinpaul/hands-on-llms): Learn about LLM, LLMOps, and vector DBS for free by designing, training, and deploying a real-time financial advisor LLM system.
17. [滑铁卢大学 CS 886: Recent Advances on Foundation Models](https://cs.uwaterloo.ca/~wenhuche/teaching/cs886/)
18. [Mistral: Getting Started with Mistral](https://www.deeplearning.ai/short-courses/getting-started-with-mistral/)
19. [斯坦福 CS25: Transformers United V4](https://web.stanford.edu/class/cs25/)
20. [Coursera: Chatgpt 应用提示工程](https://www.coursera.org/learn/prompt-engineering)
21. [LangGPT](https://github.com/langgptai/LangGPT): Empowering everyone to become a prompt expert!
22. [mistralai-cookbook](https://github.com/mistralai/cookbook)
23. [Introduction to Generative AI 2024 Spring](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)
24. [build nanoGPT](https://github.com/karpathy/build-nanogpt): Video+code lecture on building nanoGPT from scratch.
25. [LLM101n](https://github.com/karpathy/LLM101n): Let's build a Storyteller.
26. [Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)
27. [LLMs From Scratch (Datawhale Version)](https://github.com/datawhalechina/llms-from-scratch-cn)
28. [OpenRAG](https://openrag.notion.site/Open-RAG-c41b2a4dcdea4527a7c1cd998e763595)
29. [通往AGI之路](https://waytoagi.feishu.cn/wiki/QPe5w5g7UisbEkkow8XcDmOpn8e)
30. [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
31. [Interactive visualization of Transformer](https://poloclub.github.io/transformer-explainer/)
32. [andysingal/llm-course](https://github.com/andysingal/llm-course)
33. [LM-class](https://lm-class.org/lectures)
34. [Google Advanced: Generative AI for Developers Learning Path](https://www.cloudskillsboost.google/paths/183)
35. [Anthropics：Prompt Engineering Interactive Tutorial](https://github.com/anthropics/courses/tree/master/prompt_engineering_interactive_tutorial/Anthropic%201P)
36. [LLMsBook](https://github.com/liucongg/LLMsBook)
37. [Large Language Model Agents](https://llmagents-learning.org/f24)
38. [Cohere LLM University](https://cohere.com/llmu)
39. [LLMs and Transformers](https://www.ambujtewari.com/LLM-fall2024/)
40. [Smol Vision](https://github.com/merveenoyan/smol-vision): Recipes for shrinking, optimizing, customizing cutting edge vision models.
41. [Multimodal RAG: Chat with Videos](https://www.deeplearning.ai/short-courses/multimodal-rag-chat-with-videos/)
42. [LLMs Interview Note](https://github.com/wdndev/llm_interview_note)
43. [RAG++ : From POC to production](https://www.wandb.courses/courses/rag-in-production): Advanced RAG course.
44. [Weights & Biases AI Academy](https://www.wandb.courses/pages/w-b-courses): Finetuning, building with LLMs, Structured outputs and more LLM courses.
45. [Prompt Engineering & AI tutorials & Resources](https://promptengineering.org/)
46. [Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer](https://www.youtube.com/watch?v=sVcwVQRHIc8)
47. [LLM Evaluation: A Complete Course](https://www.comet.com/site/llm-course/)
48. [HuggingFace Learn](https://huggingface.co/learn)
49. [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 教程 Tutorial

1. [动手学大模型应用开发](https://datawhalechina.github.io/llm-universe/#/)
2. [AI开发者频道](https://techdiylife.github.io/blog/blog_list.html)
3. [B站：五里墩茶社](https://space.bilibili.com/615957867/?spm_id_from=333.999.0.0)
4. [B站：木羽Cheney](https://space.bilibili.com/3537113897241540/?spm_id_from=333.999.0.0)
5. [YTB：AI Anytime](https://www.youtube.com/channel/UC-zVytOQB62OwMhKRi0TDvg)
6. [B站：漆妮妮](https://space.bilibili.com/1262370256/?spm_id_from=333.999.0.0)
7. [Prompt Engineering Guide](https://www.promptingguide.ai/)
8. [YTB: AI超元域](https://www.youtube.com/@AIsuperdomain)
9. [B站：TechBeat人工智能社区](https://space.bilibili.com/209732435)
10. [B站：黄益贺](https://space.bilibili.com/322961825)
11. [B站：深度学习自然语言处理](https://space.bilibili.com/507524288)
12. [LLM Visualization](https://bbycroft.net/llm)
13. [知乎: 原石人类](https://www.zhihu.com/people/zhang-shi-tou-88-98/posts)
14. [B站：小黑黑讲AI](https://space.bilibili.com/1963375439/?spm_id_from=333.999.0.0)
15. [B站：面壁的车辆工程师](https://space.bilibili.com/669720247/?spm_id_from=333.999.0.0)
16. [B站：AI老兵文哲](https://space.bilibili.com/472543316/?spm_id_from=333.999.0.0)
17. [Large Language Models (LLMs) with Colab notebooks](https://mlabonne.github.io/blog/)
18. [YTB：IBM Technology](https://www.youtube.com/@IBMTechnology)
19. [YTB: Unify Reading Paper Group](https://www.youtube.com/playlist?list=PLwNuX3xB_tv91QvDXlW2TjrLGHW51uMul)
20. [Chip Huyen](https://huyenchip.com/blog/)
21. [How Much VRAM](https://github.com/AlexBodner/How_Much_VRAM)
22. [Blog: 科学空间（苏剑林）](https://kexue.fm/)
23. [YTB: Hyung Won Chung](https://www.youtube.com/watch?v=dbo3kNKPaUA)
24. [Blog: Tejaswi kashyap](https://medium.com/@tejaswi_kashyap)
25. [Blog: 小昇的博客](https://xiaosheng.blog/)
26. [知乎: ybq](https://www.zhihu.com/people/ybq-29-32/posts)
27. [W&B articles](https://wandb.ai/fully-connected)
28. [Huggingface Blog](https://huggingface.co/blog/zh)
29. [Blog: GbyAI](https://gby.ai/)
30. [Blog: mlabonne](https://mlabonne.github.io/blog/)
31. [LLM-Action](https://github.com/liguodongiot/llm-action)
32. [Blog: Lil’Log (OponAI)](https://lilianweng.github.io/)
33. [B站: 毛玉仁](https://space.bilibili.com/3546823125895398)
34. [AI-Guide-and-Demos](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 论文 Paper

> [!NOTE]
> 🤝[Huggingface Daily Papers](https://huggingface.co/papers)、[Cool Papers](https://papers.cool/)、[ML Papers Explained](https://github.com/dair-ai/ML-Papers-Explained)

1. [Hermes-3-Technical-Report](https://nousresearch.com/wp-content/uploads/2024/08/Hermes-3-Technical-Report.pdf)
2. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
3. [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
4. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
5. [Qwen2-vl Technical Report](https://arxiv.org/abs/2409.12191)
6. [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)
7. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
8. [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)
9. [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)
10. [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
11. [MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series](https://arxiv.org/abs/2405.19327)
12. [Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model](https://arxiv.org/abs/2404.04167)
13. [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)
14. [Jamba-1.5: Hybrid Transformer-Mamba Models at Scale](https://arxiv.org/abs/2408.12570v1)
15. [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
16. [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
17. [Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models](https://arxiv.org/abs/2408.02085) `data`
18. [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
19. [Model Merging Paper](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66)
20. [Baichuan-Omni Technical Report](https://arxiv.org/abs/2410.08565)
21. [1.5-Pints Technical Report: Pretraining in Days, Not Months – Your Language Model Thrives on Quality Data](https://arxiv.org/abs/2408.03506)
22. [Baichuan Alignment Technical Report](https://arxiv.org/abs/2410.14940v1)
23. [Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent](https://arxiv.org/abs/2411.02265)
24. [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://arxiv.org/abs/2409.17146)
25. [TÜLU 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
26. [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)
27. [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling](https://arxiv.org/abs/2412.05271)
28. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
29. [YuLan-Mini: An Open Data-efficient Language Model](https://arxiv.org/abs/2412.17743)
30. [An Introduction to Vision-Language Modeling](https://arxiv.org/abs/2405.17247)
31. [DeepSeek V3 Technical Report](https://github.com/WangRongsheng/awesome-LLM-resourses/blob/main/docs/DeepSeek_V3.pdf)
32. [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
33. [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)
34. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1)
35. [KIMI K1.5](https://github.com/WangRongsheng/awesome-LLM-resourses/blob/main/docs/Kimi_k1.5.pdf)
36. [Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models](https://arxiv.org/abs/2501.14818)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## 社区 Community

1. [魔乐社区](https://modelers.cn/)
2. [HuggingFace](https://huggingface.co/)
3. [ModelScope](https://modelscope.cn/)
4. [WiseModel](https://www.wisemodel.cn/)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## Open o1

> [!NOTE]
> 
> 开放的技术是我们永恒的追求

1. https://github.com/atfortes/Awesome-LLM-Reasoning
2. https://github.com/hijkzzz/Awesome-LLM-Strawberry
3. https://github.com/wjn1996/Awesome-LLM-Reasoning-Openai-o1-Survey
4. https://github.com/srush/awesome-o1
5. https://github.com/open-thought/system-2-research
6. https://github.com/ninehills/blog/issues/121
7. https://github.com/OpenSource-O1/Open-O1
8. https://github.com/GAIR-NLP/O1-Journey
9. https://github.com/marlaman/show-me
10. https://github.com/bklieger-groq/g1
11. https://github.com/Jaimboh/Llamaberry-Chain-of-Thought-Reasoning-in-AI
12. https://github.com/pseudotensor/open-strawberry
13. https://huggingface.co/collections/peakji/steiner-preview-6712c6987110ce932a44e9a6
14. https://github.com/SimpleBerry/LLaMA-O1
15. https://huggingface.co/collections/Skywork/skywork-o1-open-67453df58e12f6c3934738d0
16. https://huggingface.co/collections/Qwen/qwq-674762b79b75eac01735070a
17. https://github.com/SkyworkAI/skywork-o1-prm-inference
18. https://github.com/RifleZhang/LLaVA-Reasoner-DPO
19. https://github.com/ADaM-BJTU
20. https://github.com/ADaM-BJTU/OpenRFT
21. https://github.com/RUCAIBox/Slow_Thinking_with_LLMs
22. https://github.com/richards199999/Thinking-Claude
23. https://huggingface.co/AGI-0/Art-v0-3B
24. https://huggingface.co/deepseek-ai/DeepSeek-R1
25. https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero
26. https://github.com/huggingface/open-r1
27. https://github.com/hkust-nlp/simpleRL-reason
28. https://github.com/Jiayi-Pan/TinyZero
29. https://github.com/baichuan-inc/Baichuan-M1-14B
30. https://github.com/EvolvingLMMs-Lab/open-r1-multimodal
31. https://github.com/open-thoughts/open-thoughts
32. Mini-R1: https://www.philschmid.de/mini-deepseek-r1
33. LLaMA-Berry: https://arxiv.org/abs/2410.02884
34. MCTS-DPO: https://arxiv.org/abs/2405.00451
35. OpenR: https://github.com/openreasoner/openr
36. https://arxiv.org/abs/2410.02725
37. LLaVA-o1: https://arxiv.org/abs/2411.10440
38. Marco-o1: https://arxiv.org/abs/2411.14405
39. OpenAI o1 report: https://openai.com/index/deliberative-alignment
40. DRT-o1: https://github.com/krystalan/DRT-o1
41. Virgo：https://arxiv.org/abs/2501.01904
42. HuatuoGPT-o1：https://arxiv.org/abs/2412.18925
43. o1 roadmap：https://arxiv.org/abs/2412.14135
44. Mulberry：https://arxiv.org/abs/2412.18319
45. https://arxiv.org/abs/2412.09413
46. https://arxiv.org/abs/2501.02497
47. Search-o1:https://arxiv.org/abs/2501.05366v1
48. https://arxiv.org/abs/2501.18585
49. https://github.com/simplescaling/s1
50. https://github.com/Deep-Agent/R1-V
51. https://github.com/StarRing2022/R1-Nature
52. https://github.com/Unakar/Logic-RL
53. https://github.com/datawhalechina/unlock-deepseek
54. https://github.com/GAIR-NLP/LIMO
55. https://github.com/Zeyi-Lin/easy-r1
56. https://github.com/jackfsuia/nanoRLHF/tree/main/examples/r1-v0

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## Small Language Model

1. https://github.com/jiahe7ay/MINI_LLM
2. https://github.com/jingyaogong/minimind
3. https://github.com/DLLXW/baby-llama2-chinese
4. https://github.com/charent/ChatLM-mini-Chinese
5. https://github.com/wdndev/tiny-llm-zh
6. https://github.com/Tongjilibo/build_MiniLLM_from_scratch
7. https://github.com/jzhang38/TinyLlama
8. https://github.com/AI-Study-Han/Zero-Chatgpt
9. https://github.com/loubnabnl/nanotron-smol-cluster ([使用Cosmopedia训练cosmo-1b](https://huggingface.co/blog/zh/cosmopedia))
10. https://github.com/charent/Phi2-mini-Chinese
11. https://github.com/allenai/OLMo
12. https://github.com/keeeeenw/MicroLlama
13. https://github.com/Chinese-Tiny-LLM/Chinese-Tiny-LLM
14. https://github.com/leeguandong/MiniLLaMA3
15. https://github.com/Pints-AI/1.5-Pints
16. https://github.com/zhanshijinwat/Steel-LLM
17. https://github.com/RUC-GSAI/YuLan-Mini
18. https://github.com/Om-Alve/smolGPT

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## Small Vision Language Model

1. https://github.com/jingyaogong/minimind-v
2. https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
3. https://github.com/AI-Study-Han/Zero-Qwen-VL
4. https://github.com/Coobiw/MPP-LLaVA
5. https://github.com/qnguyen3/nanoLLaVA
6. https://github.com/TinyLLaVA/TinyLLaVA_Factory
7. https://github.com/ZhangXJ199/TinyLLaVA-Video
8. https://github.com/Emericen/tiny-qwen

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

## Tips

1. [What We Learned from a Year of Building with LLMs (Part I)](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)
2. [What We Learned from a Year of Building with LLMs (Part II)](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/)
3. [What We Learned from a Year of Building with LLMs (Part III): Strategy](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-iii-strategy/)
4. [轻松入门大语言模型（LLM）](https://www.bilibili.com/video/BV1pF4m1V7FB/?spm_id_from=333.999.0.0&vd_source=c739db1ebdd361d47af5a0b8497417db)
5. [LLMs for Text Classification: A Guide to Supervised Learning](https://www.striveworks.com/blog/llms-for-text-classification-a-guide-to-supervised-learning)
6. [Unsupervised Text Classification: Categorize Natural Language With LLMs](https://www.striveworks.com/blog/unsupervised-text-classification-how-to-use-llms-to-categorize-natural-language-data)
7. [Text Classification With LLMs: A Roundup of the Best Methods](https://www.striveworks.com/blog/text-classification-with-llms-a-roundup-of-the-best-methods)
8. [LLM Pricing](https://docs.google.com/spreadsheets/d/18GHPEBJzDbICmMStPVkNWA_hQHiWmLcqUdEJA1b4MJM/edit?gid=0#gid=0)
9. [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration)
10. [Tiny LLM Universe](https://github.com/datawhalechina/tiny-universe)
11. [Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
12. [Zero-Qwen-VL](https://github.com/AI-Study-Han/Zero-Qwen-VL)
13. [finetune-Qwen2-VL](https://github.com/zhangfaen/finetune-Qwen2-VL)
14. [MPP-LLaVA](https://github.com/Coobiw/MPP-LLaVA)
15. [build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
16. [Tiny LLM zh](https://github.com/wdndev/tiny-llm-zh)
17. [MiniMind](https://github.com/jingyaogong/minimind): 3小时完全从0训练一个仅有26M的小参数GPT，最低仅需2G显卡即可推理训练.
18. [LLM-Travel](https://github.com/Glanvery/LLM-Travel): 致力于深入理解、探讨以及实现与大模型相关的各种技术、原理和应用
19. [Knowledge distillation: Teaching LLM's with synthetic data](https://wandb.ai/byyoung3/ML_NEWS3/reports/Knowledge-distillation-Teaching-LLM-s-with-synthetic-data--Vmlldzo5MTMyMzA2)
20. [Part 1: Methods for adapting large language models](https://ai.meta.com/blog/adapting-large-language-models-llms/)
21. [Part 2: To fine-tune or not to fine-tune](https://ai.meta.com/blog/when-to-fine-tune-llms-vs-other-techniques/)
22. [Part 3: How to fine-tune: Focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/)
23. [Reader-LM: Small Language Models for Cleaning and Converting HTML to Markdown](https://jina.ai/news/reader-lm-small-language-models-for-cleaning-and-converting-html-to-markdown/?nocache=1)
24. [LLMs应用构建一年之心得](https://iangyan.github.io/2024/09/08/building-with-llms-part-1/)
25. [LLM训练-pretrain](https://zhuanlan.zhihu.com/p/718354385)
26. [pytorch-llama](https://github.com/hkproj/pytorch-llama): LLaMA 2 implemented from scratch in PyTorch.
27. [Preference Optimization for Vision Language Models with TRL](https://huggingface.co/blog/dpo_vlm) 【[support model](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForVision2Seq)】
28. [Fine-tuning visual language models using SFTTrainer](https://huggingface.co/blog/vlms) 【[docs](https://huggingface.co/docs/trl/sft_trainer#extending-sfttrainer-for-vision-language-models)】
29. [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
30. [Role-Playing in Large Language Models like ChatGPT](https://promptengineering.org/role-playing-in-large-language-models-like-chatgpt/)
31. [Distributed Training Guide](https://github.com/LambdaLabsML/distributed-training-guide): Best practices & guides on how to write distributed pytorch training code.
32. [Chat Templates](https://hf-mirror.com/blog/chat-templates)
33. [Top 20+ RAG Interview Questions](https://www.analyticsvidhya.com/blog/2024/04/rag-interview-questions/)
34. [LLM-Dojo 开源大模型学习场所，使用简洁且易阅读的代码构建模型训练框架](https://github.com/mst272/LLM-Dojo)
35. [o1 isn’t a chat model (and that’s the point)](https://www.latent.space/p/o1-skill-issue)
36. [Beam Search快速理解及代码解析](https://www.cnblogs.com/nickchen121/p/15499576.html)
37. [基于 transformers 的 generate() 方法实现多样化文本生成：参数含义和算法原理解读](https://blog.csdn.net/muyao987/article/details/125917234)

<div align="right">
    <b><a href="#Contents">↥ back to top</a></b>
</div>

![](https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67)

如果你觉得本项目对你有帮助，欢迎引用：
```bib
@misc{wang2024llm,
      title={awesome-LLM-resourses}, 
      author={Rongsheng Wang},
      year={2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/WangRongsheng/awesome-LLM-resourses}},
}
```

[![Forkers repo roster for @WangRongsheng/awesome-LLM-resourses](https://reporoster.com/forks/WangRongsheng/awesome-LLM-resourses)](https://github.com/WangRongsheng/awesome-LLM-resourses/network/members)

[![Stargazers repo roster for @WangRongsheng/awesome-LLM-resourses](https://reporoster.com/stars/WangRongsheng/awesome-LLM-resourses)](https://github.com/WangRongsheng/awesome-LLM-resourses/stargazers)

[![Stargazers over time](https://starchart.cc/WangRongsheng/awesome-LLM-resourses.svg)](https://starchart.cc/WangRongsheng/awesome-LLM-resourses)


