![](./assets/logo2.png)

<p align="center">全世界最好的中文大语言模型资源汇总 持续更新</p>

 <p align="center">
<a href="https://info.flagcounter.com/U6Yn"><img src="https://s11.flagcounter.com/count2/U6Yn/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_1/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
</p>

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/forks/WangRongsheng/awesome-LLM-resourses.svg?style=social >
  <img src=https://img.shields.io/github/stars/WangRongsheng/awesome-LLM-resourses.svg?style=social >
  <img src=https://img.shields.io/github/watchers/WangRongsheng/awesome-LLM-resourses.svg?style=social >
 </div>   

#### Contents

- [数据 Data](#数据-Data)
- [微调 Fine-Tuning](#微调-Fine-Tuning)
- [推理 Inference](#推理-Inference)
- [评估 Evaluation](#评估-Evaluation)
- [体验 Usage](#体验-Usage)
- [RAG](#RAG)
- [Agents](#Agents)
- [搜索 Search](#搜索-Search)
- [书籍 Book](#书籍-Book)
- [课程 Course](#课程-Course)
- [教程 Tutorial](#教程-Tutorial)
- [论文 Paper](#论文-Paper)
- [Tips](#tips)


## 数据 Data

1. [AotoLabel](https://github.com/refuel-ai/autolabel): Label, clean and enrich text datasets with LLMs.
2. [LabelLLM](https://github.com/opendatalab/LabelLLM): The Open-Source Data Annotation Platform.
3. [data-juicer](https://github.com/modelscope/data-juicer): A one-stop data processing system to make data higher-quality, juicier, and more digestible for LLMs!
4. [OmniParser](https://github.com/jf-tech/omniparser): a native Golang ETL streaming parser and transform library for CSV, JSON, XML, EDI, text, etc.
5. [MinerU](https://github.com/opendatalab/MinerU): MinerU is a one-stop, open-source, high-quality data extraction tool, supports PDF/webpage/e-book extraction.
6. [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit): A Comprehensive Toolkit for High-Quality PDF Content Extraction.
7. [Parsera](https://github.com/raznem/parsera): Lightweight library for scraping web-sites with LLMs.
8. [Sparrow](https://github.com/katanaml/sparrow): Sparrow is an innovative open-source solution for efficient data extraction and processing from various documents and images.

## 微调 Fine-Tuning

1. [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Unify Efficient Fine-Tuning of 100+ LLMs.
2. [unsloth](https://github.com/unslothai/unsloth): 2-5X faster 80% less memory LLM finetuning.
3. [TRL](https://huggingface.co/docs/trl/index): Transformer Reinforcement Learning.
4. [Firefly](https://github.com/yangjianxin1/Firefly): Firefly: 大模型训练工具，支持训练数十种大模型
5. [Xtuner](https://github.com/InternLM/xtuner): An efficient, flexible and full-featured toolkit for fine-tuning large models.
6. [torchtune](https://github.com/pytorch/torchtune): A Native-PyTorch Library for LLM Fine-tuning.
7. [Swift](https://github.com/modelscope/swift): Use PEFT or Full-parameter to finetune 200+ LLMs or 15+ MLLMs.
8. [AutoTrain](https://huggingface.co/autotrain): A new way to automatically train, evaluate and deploy state-of-the-art Machine Learning models.
9. [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework (Support 70B+ full tuning & LoRA & Mixtral & KTO).
10. [Ludwig](https://github.com/ludwig-ai/ludwig): Low-code framework for building custom LLMs, neural networks, and other AI models.
11. [mistral-finetune](https://github.com/mistralai/mistral-finetune): A light-weight codebase that enables memory-efficient and performant finetuning of Mistral's models.
12. [aikit](https://github.com/sozercan/aikit): Fine-tune, build, and deploy open-source LLMs easily!
13. [H2O-LLMStudio](https://github.com/h2oai/h2o-llmstudio): H2O LLM Studio - a framework and no-code GUI for fine-tuning LLMs.
14. [LitGPT](https://github.com/Lightning-AI/litgpt): Pretrain, finetune, deploy 20+ LLMs on your own data. Uses state-of-the-art techniques: flash attention, FSDP, 4-bit, LoRA, and more.
15. [LLMBox](https://github.com/RUCAIBox/LLMBox): A comprehensive library for implementing LLMs, including a unified training pipeline and comprehensive model evaluation.
16. [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP): Easy-to-use and powerful NLP and LLM library.
17. [workbench-llamafactory](https://github.com/NVIDIA/workbench-llamafactory): This is an NVIDIA AI Workbench example project that demonstrates an end-to-end model development workflow using Llamafactory.
18. [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework (70B+ PPO Full Tuning & Iterative DPO & LoRA & Mixtral).
19. [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory): A Framework of Small-scale Large Multimodal Models.
20. [LLM-Foundry](https://github.com/mosaicml/llm-foundry): LLM training code for Databricks foundation models.
21. [lmms-finetune](https://github.com/zjysteven/lmms-finetune): A unified codebase for finetuning (full, lora) large multimodal models, supporting llava-1.5, qwen-vl, llava-interleave, llava-next-video, phi3-v etc.
22. [Simplifine](https://github.com/simplifine-llm/Simplifine): Simplifine lets you invoke LLM finetuning with just one line of code using any Hugging Face dataset or model.
23. [Transformer Lab](https://github.com/transformerlab/transformerlab-app): Open Source Application for Advanced LLM Engineering: interact, train, fine-tune, and evaluate large language models on your own computer.
24. [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Efficient Triton Kernels for LLM Training.

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

## 评估 Evaluation

1. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models.
2. [opencompass](https://github.com/open-compass/opencompass): OpenCompass is an LLM evaluation platform, supporting a wide range of models (Llama3, Mistral, InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.
3. [llm-comparator](https://github.com/PAIR-code/llm-comparator): LLM Comparator is an interactive data visualization tool for evaluating and analyzing LLM responses side-by-side, developed.

## 体验 Usage

1. [LMSYS Chatbot Arena: Benchmarking LLMs in the Wild](https://arena.lmsys.org/)
2. [CompassArena 司南大模型竞技场](https://modelscope.cn/studios/opencompass/CompassArena/summary)
3. [琅琊榜](https://langyb.com/)
4. [Huggingface Spaces](https://huggingface.co/spaces)
5. [WiseModel Spaces](https://wisemodel.cn/spaces)
6. [Poe](https://poe.com/)
7. [林哥的大模型野榜](https://lyihub.com/)
8. [OpenRouter](https://openrouter.ai/)

## RAG

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

## Agents

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
18. [CAMEL](https://github.com/camel-ai/camel): Finding the Scaling Law of Agents. A multi-agent framework.
19. [PraisonAI](https://github.com/MervinPraison/PraisonAI/): PraisonAI application combines AutoGen and CrewAI or similar frameworks into a low-code solution for building and managing multi-agent LLM systems, focusing on simplicity, customisation, and efficient human-agent collaboration.
20. [IoA](https://github.com/openbmb/ioa): An open-source framework for collaborative AI agents, enabling diverse, distributed agents to team up and tackle complex tasks through internet-like connectivity.
21. [llama-agentic-system ](https://github.com/meta-llama/llama-agentic-system): Agentic components of the Llama Stack APIs.
22. [Agent Zero](https://github.com/frdel/agent-zero): Agent Zero is not a predefined agentic framework. It is designed to be dynamic, organically growing, and learning as you use it.
23. [Agents](https://github.com/aiwaves-cn/agents): An Open-source Framework for Data-centric, Self-evolving Autonomous Language Agents.
24. [AgentScope](https://github.com/modelscope/agentscope): Start building LLM-empowered multi-agent applications in an easier way.

## 搜索 Search

1. [OpenSearch GPT](https://github.com/supermemoryai/opensearch-ai): SearchGPT / Perplexity clone, but personalised for you.
2. [MindSearch](https://github.com/InternLM/MindSearch): An LLM-based Multi-agent Framework of Web Search Engine (like Perplexity.ai Pro and SearchGPT).
3. [nanoPerplexityAI](https://github.com/Yusuke710/nanoPerplexityAI): The simplest open-source implementation of perplexity.ai.

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

## 课程 Course

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

## 论文 Paper

> [!NOTE]
> 🤝[Huggingface Daily Papers](https://huggingface.co/papers)、[Cool Papers](https://papers.cool/)

1. [Hermes-3-Technical-Report](https://nousresearch.com/wp-content/uploads/2024/08/Hermes-3-Technical-Report.pdf)
2. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
3. [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
4. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
5. [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)
6. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
7. [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)
8. [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)
9. [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
10. [MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series](https://arxiv.org/abs/2405.19327)
11. [Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model](https://arxiv.org/abs/2404.04167)
12. [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)
13. [Jamba-1.5: Hybrid Transformer-Mamba Models at Scale](https://arxiv.org/abs/2408.12570v1)
14. [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
15. [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
16. [Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models](https://arxiv.org/abs/2408.02085) `data`
17. [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)

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

<div align="right">
    <b><a href="#pkslow-samples">↥ back to top</a></b>
</div>

<!--
## 软件
1. [POT](https://github.com/pot-app/pot-desktop)
2. [Bob](https://github.com/ripperhe/Bob)
3. [OpenAI Translator Bob Plugin](https://github.com/openai-translator/bob-plugin-openai-translator)
4. [沉浸式翻译](https://immersivetranslate.com/)
-->

[![Forkers repo roster for @WangRongsheng/awesome-LLM-resourses](https://reporoster.com/forks/WangRongsheng/awesome-LLM-resourses)](https://github.com/WangRongsheng/awesome-LLM-resourses/network/members)

[![Stargazers repo roster for @WangRongsheng/awesome-LLM-resourses](https://reporoster.com/stars/WangRongsheng/awesome-LLM-resourses)](https://github.com/WangRongsheng/awesome-LLM-resourses/stargazers)

[![Stargazers over time](https://starchart.cc/WangRongsheng/awesome-LLM-resourses.svg)](https://starchart.cc/WangRongsheng/awesome-LLM-resourses)


