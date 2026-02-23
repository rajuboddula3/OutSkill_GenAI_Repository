# GenAIEngineering-Cohort1 — Project Reference (CLAUDE.md)

## Project Overview

A comprehensive 18-module GenAI Engineering curriculum covering Python fundamentals through advanced multi-agent AI systems. Designed as a cohort-based learning program with hands-on projects, progressive RAG pipelines, multimodal AI, agent frameworks, and production deployment patterns.

| Stat | Value |
|------|-------|
| Total Modules | 15 Weeks + 2 BaseCamps + 1 BuildWeek |
| Jupyter Notebooks | 100+ |
| Python Scripts | 50+ |
| Data Files | 15+ CSV/JSON/text |
| Virtual Environments | 3 included (BaseCamp1, BaseCamp2, Week1) |

---

## Directory Structure

```
GenAIEngineering-Cohort1/
├── BaseCamp1/              # Python fundamentals (Day1, Day2)
├── BaseCamp2/              # OOP, design patterns, Streamlit (Day1, Day2)
├── BuildWeekFineTuning/    # Fine-tuning models (Day1)
├── Week1/                  # Pandas, NLP, web crawling, search
├── Week2/                  # ML models, text generation, image diffusion
├── Week3/                  # Coding assistants, Groq API
├── Week4/                  # RAG systems introduction
├── Week5/                  # Discord bots, advanced RAG
├── Week6/                  # Data integration, function calling
├── Week7/                  # Multimodal AI foundations
├── Week8/                  # Embeddings, fusion, diffusion models
├── Week9/                  # Vector databases, LanceDB, complete RAG UI
├── Week10/                 # Chatbots, REST APIs, MCP protocol
├── Week11/                 # CrewAI multi-agent framework
├── Week12/                 # Advanced agents, knowledge graphs, financial modeling
├── Week13/                 # LangGraph workflow orchestration
├── Week14/                 # Code review agents, async RAG, advanced MCP
├── Week15/                 # Batch processing, synthetic data, statistics
├── classroom.ipynb         # Main classroom notebook (74MB)
├── requirements.txt        # Root-level dependencies
├── commands.md             # FastAPI/Streamlit/Git command reference
└── .gitignore
```

---

## Week-by-Week Summary

| Module | Topics | Key Technologies |
|--------|--------|-----------------|
| BaseCamp1 | Python basics, OOP, file I/O, NumPy, Pandas, threading | Python 3.11, Jupyter, Groq API |
| BaseCamp2 | OOP patterns, calculator evolution, Streamlit web apps | Streamlit, Python OOP |
| BuildWeek | Fine-tuning introduction (Shakespeare corpus) | Transformers |
| Week 1 | Pandas, NLP n-grams, web crawling, search engine | pandas, NLTK, Streamlit |
| Week 2 | Time series, classification, regression, sentiment, diffusion, QA | scikit-learn, transformers, HuggingFace |
| Week 3 | AI coding assistants, Groq LLM integration | Groq API |
| Week 4 | RAG basics, retrieval pipelines | LangChain, Qdrant |
| Week 5 | Discord bots, advanced RAG, multi-turn chat | nextcord, LangChain |
| Week 6 | MySQL/CSV/JSON RAG, hybrid search, function calling, structured output | MySQL, LangChain, Qdrant |
| Week 7 | Multimodal data representation, preprocessing, BLEU/CLIP metrics | transformers, librosa, Pillow |
| Week 8 | Text/image/audio embeddings, CLIP, BLIP, Stable Diffusion | Sentence Transformers, ViT, Wav2Vec2, diffusers |
| Week 9 | LanceDB vector DB, multimodal search, Gradio UI, full RAG pipeline | LanceDB, Gradio, CLIP, Qwen |
| Week 10 | Multi-turn chatbot, quiz generator, REST APIs, MCP stdio/SSE | FastAPI, Streamlit, MCP, OpenRouter |
| Week 11 | CrewAI agents/tasks/tools, annual reports, news aggregator, SDLC planner | CrewAI, FAISS |
| Week 12 | Advanced agents, structured output, knowledge graphs, A2A communication, financial modeling | CrewAI, PhiData, LangGraph |
| Week 13 | LangGraph state graphs, story gen, support tickets, blog pipeline | LangGraph |
| Week 14 | Code review agents, async RAG, advanced MCP patterns | LangGraph, asyncio, MCP |
| Week 15 | Batch APIs, image generation, synthetic data, statistical distributions | Claude API, diffusers, scipy |

---

## Technologies & Libraries

### Core AI / ML
| Library | Purpose |
|---------|---------|
| `transformers` | HuggingFace transformer models |
| `torch` | Deep learning framework |
| `sentence-transformers` | Embedding models |
| `scikit-learn` | Classical ML algorithms |
| `langchain-community` | LangChain integrations |
| `langchain-huggingface` | HuggingFace LangChain bridge |
| `langchain-qdrant` | Qdrant vector DB integration |
| `crewai` / `crewai_tools` | Multi-agent framework |
| `langgraph` | Graph-based agent workflows |
| `phi` / `phidata` | Alternative agent framework |
| `ollama` | Local LLM runtime |

### Vector Databases & Search
| Library | Purpose |
|---------|---------|
| `qdrant-client` | Qdrant vector database |
| `lancedb` | LanceDB vector database |
| `faiss-cpu` | Facebook AI similarity search |
| `rank_bm25` | BM25 keyword ranking |

### APIs & Integrations
| API | Weeks Used | Env Var |
|-----|-----------|---------|
| Google Generative AI | 1–6 | `GOOGLE_API_KEY` |
| Groq API | BaseCamp1, Week3 | `GROQ_API_KEY` |
| HuggingFace | Week2–8 | `HF_TOKEN` |
| OpenRouter | Week10–14 | `OPENROUTER_API_KEY` |
| OpenAI (optional) | Week9–10 | `OPENAI_API_KEY` |
| Discord Bot API | Week5 | `DISCORD_TOKEN` |

### Web Frameworks
| Library | Purpose |
|---------|---------|
| `fastapi` | REST API development |
| `streamlit` | Rapid web app prototyping |
| `gradio` | ML model interfaces |
| `uvicorn` | ASGI server for FastAPI |
| `aiohttp` | Async HTTP client |

### Data & Visualization
| Library | Purpose |
|---------|---------|
| `pandas` | DataFrame operations |
| `numpy<2` | Numerical computing (pinned version) |
| `matplotlib` | Static visualizations |
| `seaborn` | Statistical plots |
| `plotly` | Interactive visualizations |

---

## BaseCamp Modules

### BaseCamp1 — Python Fundamentals
**Path**: `BaseCamp1/`
**Environment**: `bascamp1_env/` (Python 3.11)

| Day | Files | Topics |
|-----|-------|--------|
| Day 1 | `1_My_First_Prog.py`, `2_First_Note_Book.ipynb` … `7_Groq.ipynb` | Python basics, Jupyter, functions, OOP, file I/O, Groq API |
| Day 2 | `1_Lists.ipynb` … `12_Process.py` | Lists, tuples, dicts, strings, loops, iterators, exceptions, NumPy, Pandas, threading, multiprocessing |

**Data files**: `Data_1.csv`, `Data_2.csv`, `sec_bhavdata_full_10032025.csv`, `summary_statistics.csv`

### BaseCamp2 — Advanced Python & Web Development
**Path**: `BaseCamp2/`
**Environment**: `basecamp2_env/` (Python 3.11)

| Day | Files | Topics |
|-----|-------|--------|
| Day 1 | `1_calculator.py` → `5a_text_service.py`, `6a_products.csv`, `6b_products.py` | OOP evolution, calculator design patterns, product management |
| Day 2 | `01_streamlit.py` → `09_streamlit.py`, `calculator.py` | Progressive Streamlit app development |

### BuildWeek — Fine-Tuning
**Path**: `BuildWeekFineTuning/Day1/`

- `shakespeare.txt` — Text corpus used for fine-tuning demonstrations

---

## Weekly Curriculum Detail

### Week 1 — Pandas, NLP & Search
**Path**: `Week1/`
**Environment**: `week1_env/` (Python 3.11)
**Requirements**: `Week1/requirements.txt`

**Day 1 Key Files:**
- `pandas.ipynb` — Pandas tutorial (currently modified in git)
- `pandas_homework.md` / `pandas_homework_solution.ipynb`
- `Advanced_Pandas_homework.md` / `Advanced_Pandas_Solutions.ipynb`
- `pandasql.ipynb` — SQL on DataFrames
- `nlp_homework.md` / `nlp_homework_solution.ipynb`
- `web_crawl.ipynb` — Web scraping
- `generate.ipynb` — N-gram text generation
- `search_engine.py` + `search_ui.py` — Search engine with Streamlit UI
- `bigram_probs.json`, `trigram_probs.json`, `unigram_probs.json` — N-gram models
- `customers.csv`, `sales.csv` — Practice data

**Day 2:**
- `github_ready_reckoner.md` — Git/GitHub workflow guide

---

### Week 2 — ML Models & Generative AI
**Path**: `Week2/`
**Setup**: Requires `HF_TOKEN` in `.env`

**Day 1:**
- `1_timeseries.ipynb` — Time series analysis
- `2_tabular_classification.ipynb` — Binary/multiclass classification
- `3_tabular_regression.ipynb` — Linear/polynomial regression
- `3.1_diabetes_regression.ipynb` — Regression case study
- `4_sentiment_classification.ipynb` — NLP sentiment
- `5_ratings_classification.ipynb` — Rating prediction
- `6_zeroshot_classification.ipynb` — Zero-shot learning
- `7_text_summarization.ipynb` — Abstractive summarization
- `8_text_generation.ipynb` — LLM-based generation
- `air_passengers_analysis.ipynb` — Time series case study

**Day 2:**
- `1_question_answer.ipynb` — QA systems
- `2_image_diffusion.ipynb` — Diffusion model overview
- `3_transformer_search.ipynb` — Semantic search
- `5_generative_ai.ipynb` — GenAI techniques
- `chatbot.py` + `chatbot_homework.md` + `chatbot_homework_solution.py`
- `search_engine.py` + `search_ui.py`

---

### Week 3 — Coding Assistants
**Path**: `Week3/`
**Setup**: Requires `GROQ_API_KEY`

- `coding_assistant.ipynb` — Interactive notebook
- `coding_assistant.py` — Python implementation
- `Homework.txt`, `Readme.md`

---

### Week 4 — RAG Introduction
**Path**: `Week4/`

- `Rag.ipynb` — RAG pipeline fundamentals
- `all_ml_ideas.csv`, `IndianFoodDataset.csv` — Practice datasets

---

### Week 5 — Discord Bots & Advanced RAG
**Path**: `Week5/`

**Bot Structure (`bot/`):**
- `bot.py` — Main bot entry point
- `cogs/ibot.py`, `cogs/meta.py`, `cogs/template.py` — Bot command modules
- `helpers/utils.py`, `helpers/views.py` — Utilities and UI
- `database/` — Database models
- `sample_config.json` — Configuration template
- `requirements.txt`, `README.md`

**Alternative bot:** `discord_bot/cogs/ibot.py`

**Other files:**
- `ChatAssistant.ipynb` — Chat system tutorial
- `RagAdvanced.ipynb` — Advanced RAG pipeline
- `IndianFoodDataset.csv`, `train.csv`

---

### Week 6 — Data Integration & Function Calling
**Path**: `Week6/`

| File | Topic |
|------|-------|
| `1_MySQL_Data.ipynb` | MySQL database integration |
| `2_Retrived_Data_Format.ipynb` | Data formatting |
| `3_Str_Data_RAG.ipynb` | Structured data RAG |
| `4_RAG_Pipeline.ipynb` | Full pipeline construction |
| `5_Refined_Pipeline.py` | Python pipeline implementation |
| `6_Data_Ingestion.ipynb` | Data loading techniques |
| `7_Hybrid_Search.ipynb` | BM25 + vector hybrid search |
| `8_CSV_Data.ipynb` | CSV RAG processing |
| `9_JSON_Data.ipynb` | JSON RAG processing |
| `10_Functon_Calling.ipynb` | LLM function calling |
| `11_UnStr_to_Str.ipynb` | Unstructured → structured conversion |

**Data files**: `call_records.csv`, `product_catalogue.csv`, `Student_Performance.csv`, `Student_Personal_Details.csv`, `survey_data.json`

---

### Week 7 — Multimodal AI Foundations
**Path**: `Week7/`

**Session 1:**
- `01-multimodal-data-representation.ipynb` — Data structures for text/image/audio
- `02-individual-modality-processing.ipynb` — Preprocessing pipelines
- `03-evaluation.ipynb` — BLEU, CLIP evaluation metrics

**Session 2:**
- `01-dataset-loading-and-preprocessing.ipynb` — Real dataset handling
- `02-data-alignment.ipynb` — Temporal synchronization across modalities

---

### Week 8 — Embeddings, Fusion & Diffusion
**Path**: `Week8/`

**Session 1 — Embeddings:**
- `text_embeddings.ipynb` — Sentence Transformers, BERT embeddings
- `image_embedding.ipynb` — Vision Transformer (ViT) embeddings
- `audio_embedding.ipynb` — Wav2Vec2 audio embeddings
- `late_fusion.ipynb` — Multimodal fusion strategies

**Session 2 — Advanced Multimodal:**
- `clip.ipynb` — CLIP for cross-modal understanding
- `blip.ipynb` — Image captioning and VQA with BLIP
- `diffusion_text2img.ipynb` — Text-to-image generation
- `diffusion_img2img.ipynb` — Image-to-image transformation
- `diffusion_inpainting.ipynb` — Image inpainting and editing

**Assignments:**
- `assignment/assignment_smart_food_image_generator.ipynb`
- `assignment/assignment_smart_product_cataloger.ipynb`
- `solutions/` — Worked solutions

---

### Week 9 — Vector Databases & Production RAG
**Path**: `Week9/`

**Session 1 — LanceDB & Multimodal Search:**
- `introduction_to_gradio.ipynb` — Gradio UI framework
- `lancedb_multimodal_myntra_fashion_search_engine.ipynb` — Fashion search with CLIP + LanceDB
- `food_app.py` — Gradio food product app
- `product_cataloger_app.py` — Product catalog UI

**Session 2 — Complete RAG Pipeline:**
- `app.py` — Gradio web interface (entry point)
- `rag_pipeline.py` — End-to-end orchestration
- `retriever.py` — Vector search management
- `generator.py` — LLM integration (Qwen, OpenAI)
- `augmenter.py` — Prompt engineering and context enhancement

**Assignment:**
- `assignment/assignment_fashion_rag.py`
- `solution/solution_fashion_rag.py`

---

### Week 10 — Chatbots, REST APIs & MCP
**Path**: `Week10/`

**Day 1 — Chatbot Applications:**
| File | Purpose |
|------|---------|
| `chatbot_app.py` | Multi-turn chatbot (Streamlit) |
| `chatbot_models.py` | Model management |
| `chatbot_openrouter.py` | OpenRouter API integration |
| `quiz_app.py` | Quiz generation system |
| `quiz_generator.py` | Quiz creation engine |
| `agent_chat_app.py` | Agent-based chat with tools |
| `agent_system_prompt.py` | Agent system prompts |
| `test_chatbot.py` | Unit tests |
| `test_agent_app.py` | Agent tests |
| `scripts/seed_db.py` | Database initialization |

**Day 2 — REST APIs & MCP:**
| File | Purpose |
|------|---------|
| `0_rest_example.ipynb` | REST concepts |
| `0_rest_weather.ipynb` | Weather API example |
| `1_rest_calculator_client.ipynb` / `server.py` | REST calculator |
| `2_mcp_calculator_stdio_client.ipynb` / `server.py` | MCP stdio transport |
| `3_mcp_calculator_sse_client.ipynb` / `server.py` | MCP SSE transport |
| `4_mcp_calculator_resource_client.ipynb` / `server.py` | MCP resources |
| `5_rest_server_mcp_client.py` | Hybrid REST+MCP |
| `5_streamlit_app.py` | Streamlit MCP UI |

---

### Week 11 — CrewAI Agent Framework
**Path**: `Week11/`

**Day 1 — CrewAI Basics:**
- `0_open_router.ipynb` — Model provider setup
- `1_crewai_agents_notebook.ipynb` — Agent definition
- `2_crewai_tasks_notebook.ipynb` — Task creation
- `3_crewai_tools_demo.ipynb` — Built-in tools
- `4_crewai_workflows_notebook.ipynb` — Workflow orchestration
- `5_crewai_custom_tools_notebook.ipynb` — Custom tool creation
- `6_crewai_callbacks_intro.ipynb` — Event callbacks

**Day 2 — Real-World Applications:**

| App | Path | Description |
|-----|------|-------------|
| Annual Reports | `1_annual_reports_analysis/` | PDF ingestion + FAISS + CrewAI analysis pipeline |
| News Aggregator | `2_news_aggregator/` | News API + CrewAI + Streamlit UI |
| SDLC Planner | `3_sdlc_plan/` | Software development lifecycle planning |
| Code Assistant | `4_code_assist/` | Code analysis and suggestion agent |

---

### Week 12 — Advanced Agents & Orchestration
**Path**: `Week12/`

**Day 1:**
- `1_structured_output.py` — Structured LLM outputs with Pydantic
- `3a_csv_example.py` — CSV data agent
- `4a_knowledge_Graph.ipynb` — Knowledge graph construction
- `6_python_agent.py` — Python code execution agent
- `7_plantUML.py` — PlantUML architecture diagrams

**Day 2 — Production Applications:**

| App | Path | Description |
|-----|------|-------------|
| Agent-to-Agent | `1_agent2agent/` | CrewAI ↔ PhiData inter-agent communication |
| Code Generator | `2_app_builder/` | Spec-driven code generation |
| Financial Modeling | `3_financial_modelling/` | Multi-agent financial analysis system |

**Financial Modeling System (`3_financial_modelling/`):**
- `main.py` — Orchestration entry point
- `agents/data_analyst.py` — Data analysis agent
- `agents/forecaster.py` — Forecasting agent
- `agents/portfolio_optimizer.py` — Portfolio optimization
- `agents/risk_assessor.py` — Risk assessment
- `agents/report_generator.py` — Report generation
- `utils/data_sources.py` — Financial data APIs
- `utils/financial_utils.py` — Financial calculations
- `data/sample_data.json` — Sample data

---

### Week 13 — LangGraph Workflow Orchestration
**Path**: `Week13/`

**Day 1:**
- `simple_graph.py` — Basic LangGraph graph construction
- `storygen.py` — Story generation workflow
- `supporticket.py` — Support ticket processing pipeline

**Day 2:**
- `codegen.py` — Code generation from specifications
- `flight.py` — Flight booking workflow
- `Blog Writer Team/blog_pipeline.py` — Blog writing multi-agent pipeline
- `Blog Writer Team/ui-agent.py` — Blog generation UI

---

### Week 14 — Advanced Patterns
**Path**: `Week14/`

**Day 1:**
- `codereview.py` — Automated code review agent
- `csv_1.py` — CSV data agent
- `test.py` — Testing utilities

**Day 2:**
- `arag.py` — Async RAG implementation
- `mcp.py` — Advanced MCP patterns

---

### Week 15 — Batch Processing & Synthetic Data
**Path**: `Week15/`

**Day 1 — Advanced Features:**
- `batch.ipynb` — Batch API usage patterns
- `image.ipynb` — Advanced image generation
- `predicted_output.ipynb` — Output prediction techniques
- `tokens.ipynb` — Token optimization

**Day 2 — Synthetic Data & Statistics:**
- `2_rules_based_generation_notebook.ipynb` — Rule-based data generation
- `3_statistical_data_types.ipynb` — Statistical foundations
- `4_probability_distributions.ipynb` — Distribution modeling
- `5_statistical_customer_dataset.ipynb` — Customer data generation
- `6_representative_dataset.ipynb` — Dataset representativeness
- `7_crew_text_synthetic_data.ipynb` — CrewAI-driven synthetic data

---

## Key Architectural Progressions

### RAG Pipeline Evolution
```
Week4: Basic RAG (LangChain + CSV)
  ↓
Week5: Advanced RAG (multi-turn context)
  ↓
Week6: Multi-source RAG (MySQL + CSV + JSON + hybrid search)
  ↓
Week9: Production RAG (LanceDB + Gradio UI + modular architecture)
  ↓
Week14: Async RAG (async retrieval + MCP integration)
```

### Agent Framework Evolution
```
Week10: Single-agent chatbot + function calling
  ↓
Week11: CrewAI multi-agent crews (sequential + hierarchical)
  ↓
Week12: Advanced orchestration (A2A communication, knowledge graphs)
  ↓
Week13: LangGraph stateful graph workflows
  ↓
Week14: Production patterns (code review, async, advanced MCP)
```

### Multimodal AI Progression
```
Week7: Fundamentals (data representation, preprocessing, evaluation)
  ↓
Week8: Embeddings + Fusion + Diffusion (CLIP, BLIP, Stable Diffusion)
  ↓
Week9: Production multimodal search (LanceDB + CLIP + Gradio)
```

---

## Data Files Inventory

| File | Location | Description |
|------|----------|-------------|
| `Data_1.csv`, `Data_2.csv` | `BaseCamp1/Day_1/` | Practice data |
| `sec_bhavdata_full_10032025.csv` | `BaseCamp1/Day_2/` | NSE/BSE market data |
| `summary_statistics.csv` | `BaseCamp1/Day_2/` | Statistical summary |
| `6a_products.csv` | `BaseCamp2/Day_1/` | Product catalog |
| `customers.csv`, `sales.csv` | `Week1/Day_1/` | CRM data |
| `bigram_probs.json` | `Week1/Day_1/` | Bigram n-gram model |
| `trigram_probs.json` | `Week1/Day_1/` | Trigram n-gram model |
| `unigram_probs.json` | `Week1/Day_1/` | Unigram probabilities |
| `all_ml_ideas.csv` | `Week4/` | ML project ideas dataset |
| `IndianFoodDataset.csv` | `Week4/`, `Week5/` | Indian food data |
| `train.csv` | `Week5/` | Training dataset |
| `call_records.csv` | `Week6/` | Customer call data |
| `product_catalogue.csv` | `Week6/` | Product catalog |
| `Student_Performance.csv` | `Week6/` | Student grades |
| `Student_Personal_Details.csv` | `Week6/` | Student demographics |
| `survey_data.json` | `Week6/` | Survey responses |
| `sample_data.json` | `Week12/Day_2/3_financial_modelling/data/` | Financial sample |
| `metadata.json` | `Week11/Day_2/1_annual_reports_analysis/faiss_index/` | FAISS index metadata |
| `shakespeare.txt` | `BuildWeekFineTuning/Day1/` | Fine-tuning corpus |

---

## Environment Setup

### API Keys Required (`.env` file per week)
```bash
GOOGLE_API_KEY=your_google_api_key       # Weeks 1-6
HF_TOKEN=your_huggingface_token          # Week 2-8
GROQ_API_KEY=your_groq_api_key           # BaseCamp1, Week 3
OPENROUTER_API_KEY=your_openrouter_key   # Weeks 10-14
OPENAI_API_KEY=your_openai_key           # Weeks 9-10 (optional)
DISCORD_TOKEN=your_discord_token         # Week 5
NEWS_API_KEY=your_news_api_key           # Week 11
```

### Virtual Environments
```bash
# BaseCamp1
source BaseCamp1/bascamp1_env/bin/activate

# BaseCamp2
source BaseCamp2/basecamp2_env/bin/activate

# Week 1
source Week1/week1_env/bin/activate

# For other weeks — create fresh env per week
python -m venv venv && source venv/bin/activate
pip install -r WeekN/requirements.txt
```

### Common Commands (from `commands.md`)
```bash
# Run Streamlit app
streamlit run app.py

# Run FastAPI server
uvicorn server:app --reload --port 8000

# Run Gradio app
python app.py

# Git workflow
git add .
git commit -m "message"
git push origin main
```

---

## Python Scripts Quick Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `search_engine.py` | Week1/Day_1/, Week2/Day_2/ | Text search implementation |
| `search_ui.py` | Week1/Day_1/, Week2/Day_2/ | Streamlit search frontend |
| `coding_assistant.py` | Week3/ | Groq-powered code assistant |
| `5_Refined_Pipeline.py` | Week6/ | Multi-source RAG pipeline |
| `bot.py` | Week5/bot/ | Discord bot entry point |
| `chatbot_app.py` | Week10/Day1/ | Multi-turn Streamlit chatbot |
| `agent_chat_app.py` | Week10/Day1/ | Tool-using agent chatbot |
| `quiz_app.py` | Week10/Day1/ | Quiz generation app |
| `1_rest_calculator_server.py` | Week10/Day2/ | FastAPI REST server |
| `2_mcp_calculator_stdio_server.py` | Week10/Day2/ | MCP stdio server |
| `3_mcp_calculator_sse_server.py` | Week10/Day2/ | MCP SSE server |
| `food_app.py` | Week9/ | Gradio food app |
| `rag_pipeline.py` | Week9/ | Modular RAG orchestrator |
| `simple_graph.py` | Week13/ | Basic LangGraph graph |
| `codereview.py` | Week14/Day1/ | Code review agent |
| `arag.py` | Week14/Day2/ | Async RAG agent |
| `main.py` | Week11/Day2/1_annual_reports_analysis/ | Annual reports pipeline |
| `news_app.py` | Week11/Day2/2_news_aggregator/ | News aggregator UI |
| `main.py` | Week12/Day2/3_financial_modelling/ | Financial modeling orchestrator |

---

## Notes for Claude Code

- **Do not alter** existing notebooks, scripts, or data files unless explicitly requested.
- Each week has its own `requirements.txt` — install per-week, not globally.
- The root `requirements.txt` covers the broadest set of dependencies (Google GenAI, LangChain, Qdrant, Streamlit, Transformers).
- `numpy<2` is pinned in root requirements — important for compatibility with older HuggingFace models.
- Week-specific `.env` files are expected but not committed (see `.gitignore`).
- `classroom.ipynb` at root is 74MB — avoid loading unless necessary.
- The `Week1/Day_1/pandas.ipynb` file has uncommitted modifications (git status shows `M`).
