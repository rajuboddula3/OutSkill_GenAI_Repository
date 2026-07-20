# GenAIEngineering-Cohort1 — Project Reference (CLAUDE.md)

## Project Overview

A comprehensive 18-module GenAI Engineering curriculum covering Python fundamentals through advanced multi-agent AI systems. Designed as a cohort-based learning program with hands-on projects, progressive RAG pipelines, multimodal AI, agent frameworks, and production deployment patterns.

| Stat | Value |
|------|-------|
| Total Modules | 15 Weeks + 2 BaseCamps + 1 BuildWeek |
| Jupyter Notebooks | 109 |
| Python Scripts | 117 |
| Data Files | 15+ CSV/JSON/text |
| Virtual Environments | 1 committed (`BaseCamp1/.venv`); create per-week for others |

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
├── vscode-extensions.txt   # VS Code extensions list (currently empty)
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
**Path**: `BuildWeekFineTuning/`

| Day | Files | Topics |
|-----|-------|--------|
| Day 1 | `shakespeare.txt`, `archive.zip`, `whiteboard.excalidraw` | Fine-tuning corpus + walkthrough whiteboard |
| Day 2 | `Day2.excalidraw` | Fine-tuning session 2 whiteboard |

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

**Day 2** (folder is `Day 2/` — note the space):
- `codegen.py` — Code generation from specifications
- `flight.py` — Flight booking workflow
- `requirements.txt` — Day 2 dependencies
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
# BaseCamp1 (the only committed virtual environment)
source BaseCamp1/.venv/bin/activate

# For all other modules — create a fresh env per week
python -m venv .venv && source .venv/bin/activate
pip install -r WeekN/requirements.txt   # if the week ships a requirements.txt
```

> Note: only `BaseCamp1/.venv` is checked in. Earlier `*_env/` folders
> (basecamp2_env, week1_env, …) are gitignored — recreate them locally as needed.

### UV Package Manager (preferred)

Several modules are managed with [**uv**](https://docs.astral.sh/uv/) — a fast Python
package/environment manager. Modules with a `pyproject.toml` + `uv.lock` +
`.python-version` are uv projects: **BaseCamp1**, **BaseCamp2**, **Week1**, **Week2**
(more added over time). Python is pinned to **3.11.4** via `.python-version`.

```bash
# Install uv (once, macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From inside a uv-managed module (e.g. BaseCamp1/ or Week1/):
uv sync                       # Create .venv + install exactly what uv.lock pins
uv run python script.py       # Run inside the project env (no manual activate)
uv run jupyter lab            # Launch Jupyter in the project env
uv run streamlit run app.py   # Run a Streamlit app in the project env

# Manage dependencies (updates pyproject.toml AND uv.lock)
uv add pandas                 # Add a runtime dependency
uv add --dev pytest           # Add a dev dependency
uv remove pandas              # Remove a dependency
uv lock                       # Re-resolve and refresh uv.lock

# Pin / switch Python version for the module
uv python pin 3.11.4          # Writes .python-version
uv python install 3.11.4      # Download a specific CPython build
uv python list                # List installed / available Pythons

# Environment & lockfile
uv venv                       # Create a bare .venv (without installing)
uv venv --python 3.11.4       # Create a .venv with a specific Python
uv sync --frozen              # Install strictly from uv.lock (fail if out of date)
uv sync --upgrade             # Upgrade deps within pyproject constraints, refresh lock
uv lock --upgrade-package pandas   # Upgrade a single package in the lock
uv tree                       # Show the dependency tree

# Start a fresh uv project (weeks that don't have one yet)
uv init                       # Create pyproject.toml + .python-version in the module

# pip-compatible interface (inside the project env)
uv pip install <pkg>          # Ad-hoc install (does NOT update pyproject/uv.lock)
uv pip list                   # List installed packages
uv pip freeze                 # Export installed versions

# One-off tools without adding them to the project
uv tool run ruff check .      # Run a tool in an ephemeral env
uvx ruff check .              # Shorthand for `uv tool run`
```

- Commit both `pyproject.toml` **and** `uv.lock` so environments stay reproducible.
- `uv sync` is the uv equivalent of `python -m venv .venv && pip install -r requirements.txt`.
- Prefer `uv add`/`uv remove` (updates the lockfile) over `uv pip install` (does not).
- Use `uv run <cmd>` instead of activating the venv — it auto-syncs and runs in the project env.
- Weeks without a `pyproject.toml` still use the plain `venv` + `requirements.txt` flow above.

### GitHub Repositories

This working copy has **two** remotes:

| Remote | Points to | Purpose |
|--------|-----------|---------|
| `origin` | `github.com/rajuboddula3/OutSkill_GenAI_Repository` | Personal fork — push your work here |
| `upstream` | `github.com/outskill-git/GenAIEngineering-Cohort1` | Canonical cohort repo — pull instructor updates from here |

```bash
# Push your work to your fork
git push origin main

# Pull the latest cohort material from upstream and merge into your main
git fetch upstream
git checkout main
git merge upstream/main
git push origin main          # keep your fork up to date

# Common GitHub CLI tasks (requires `gh auth login`)
gh repo view                  # Show current repo
gh pr create                  # Open a pull request
gh pr status                  # PRs relevant to you
```

> ⚠️ **Security:** do not embed a personal access token (`ghp_…`) directly in the
> remote URL — it gets stored in `.git/config` in plaintext. Prefer `gh auth login`,
> a credential helper, or SSH. To scrub a token from an existing remote:
> `git remote set-url origin https://github.com/rajuboddula3/OutSkill_GenAI_Repository.git`
> and rotate the leaked token in **GitHub → Settings → Developer settings → Personal access tokens**.

### Common Commands (from `commands.md`)
```bash
# Run Streamlit app
streamlit run app.py

# Run FastAPI server
uvicorn server:app --reload --port 8000

# Run Gradio app
python app.py
```

### Troubleshooting: "address already in use" (Errno 48)

When starting a server (FastAPI/uvicorn, MCP SSE, Gradio, Streamlit) you may see:

```
ERROR:    [Errno 48] error while attempting to bind on address ('0.0.0.0', 9321): address already in use
```

This means a previous instance of the server is still running and holding the
port (common when a dev server was closed by shutting the terminal tab instead
of pressing **Ctrl+C**). Identify the process and free the port:

```bash
# 1. Find what is listening on the port (replace 9321 with your port)
lsof -nP -iTCP:9321 -sTCP:LISTEN

# 2. Inspect the offending process before killing (optional sanity check)
ps -p <PID> -o pid,ppid,etime,command

# 3. Stop it — try a graceful stop first
kill <PID>

# 4. If it survives (uvicorn's --reload can ignore SIGTERM), force-kill
kill -9 <PID>

# 5. Confirm the port is free
lsof -nP -iTCP:9321 -sTCP:LISTEN   # (no output = free)
```

One-liner to kill whatever is on a port:

```bash
kill -9 $(lsof -tiTCP:9321 -sTCP:LISTEN)
```

Or by process name (e.g. a stray calculator server):

```bash
pkill -f 2a_calculator.py
```

**Avoid it next time:** stop servers with **Ctrl+C** in their terminal, or run
the new instance on a different port (e.g. `--port 9322`).

> macOS note: `timeout` is not installed by default — use Ctrl+C or the
> `kill`/`pkill` commands above instead.

### Troubleshooting: pandas 3.x migration gotchas

`BaseCamp2/.venv` currently runs **pandas 3.0.3 / numpy 2.4.6**. Several long-deprecated
APIs were **removed** in pandas 3.0, so older cohort notebooks and scripts written against
pandas 1.x/2.x can fail with `AttributeError`. Verified against the BaseCamp2 env:

| Old (removed) | New (use this) | Notes |
|---------------|----------------|-------|
| `df.applymap(fn)` | `df.map(fn)` | Element-wise over a DataFrame |
| `df.style.applymap(fn)` | `df.style.map(fn)` | Element-wise styling |
| `df.style.applymap_index(fn)` | `df.style.map_index(fn)` | Element-wise index styling |
| `df.append(other)` | `pd.concat([df, other])` | Removed back in pandas 2.0 |

`DataFrame.apply` and `Styler.apply` are **unchanged** — only the `applymap` family was
renamed. The rename is purely cosmetic: `.map` has identical element-wise semantics.

```python
# AttributeError: 'Styler' object has no attribute 'applymap'
st.dataframe(df.style.applymap(highlight, subset=['status']))   # pandas < 3
st.dataframe(df.style.map(highlight, subset=['status']))        # pandas >= 3  ✅
```

Other pandas 3.0 behavior changes to be aware of:

- **Copy-on-Write is always on** and can no longer be disabled. Chained assignment
  (`df[df.a > 1]['b'] = 0`) silently does nothing — use `.loc` instead:
  `df.loc[df.a > 1, 'b'] = 0`.
- Setting `pd.options.mode.copy_on_write` now raises a `Pandas4Warning` and has no
  effect; remove any such lines.

```bash
# Find affected call sites across the repo before running old material
# NOTE: quote the --include globs — unquoted, zsh tries to expand them and errors
#       with "no matches found: --include=*.py"
grep -rn "applymap" --include="*.py" --include="*.ipynb" .
```

As of the last sweep this returns **no matches** — the repo is clean of `applymap`.

> ⚠️ **numpy version conflict:** the root `requirements.txt` pins `numpy<2` (needed by
> some older HuggingFace/transformers material), but BaseCamp2's `pyproject.toml`
> requires `numpy>=2.4.6`. These are incompatible — keep them in **separate
> environments** (per-module `.venv`) rather than installing both into one.

---

## Git Command Reference

Full reference lives in `commands.md`. Quick reference for day-to-day cohort work:

### Everyday workflow
```bash
git status                       # See what changed
git add <file>                   # Stage a specific file
git add .                        # Stage everything in the current directory
git add -A                       # Stage everything in the repo
git commit -m "message"          # Commit staged changes
git commit -am "message"         # Stage tracked files + commit in one step
git push origin main             # Push commits to the remote main branch
git pull origin main             # Fetch + merge latest from remote main
```

### First-time setup
```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git clone https://github.com/username/repository.git
git clone -b <branch> https://github.com/username/repository.git   # specific branch
```

### Branching
```bash
git branch                       # List local branches
git branch -a                    # List all branches (local + remote)
git checkout -b <branch>         # Create and switch to a new branch
git checkout <branch>            # Switch to an existing branch
git merge <branch>               # Merge <branch> into the current branch
git branch -d <branch>           # Delete a merged branch
```

### Remotes
```bash
git remote -v                                        # List remotes
git remote add origin <url>                          # Add a remote
git remote set-url origin <new-url>                  # Change remote URL
git fetch origin                                     # Fetch without merging
git push -u origin <branch>                          # Push and set upstream tracking
git push --force-with-lease                          # Safer force push
```

### Inspecting history & changes
```bash
git log --oneline                # Compact history
git log --graph --oneline        # Branch graph
git diff                         # Unstaged changes
git diff --staged                # Staged changes
git show <commit-hash>           # Changes in a specific commit
git blame <file>                 # Who changed each line
```

### Undo & recovery
```bash
git restore <file>               # Discard working-directory changes (Git 2.23+)
git restore --staged <file>      # Unstage a file (keeps edits)
git reset --soft  <commit>       # Move HEAD, keep changes staged
git reset --hard  <commit>       # Discard all changes (use with caution)
git revert <commit>              # New commit that undoes a previous one
git stash / git stash pop        # Shelve and restore work-in-progress
git clean -fd                    # Remove untracked files & directories
```

### Sync a fork with upstream
```bash
git remote add upstream https://github.com/original/repository.git
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## Running & Executing Programs

Every module falls into one of a few program **types**. Learn the run pattern for
the type, then use the per-module tables below for the exact command + key packages.
Install dependencies first — a module's `requirements.txt`/`pyproject.toml` is the
source of truth; the "Key packages" column just names the headline ones.

### How to install dependencies (pick one)

```bash
# uv-managed modules (BaseCamp1, BaseCamp2, Week1, Week2 — have pyproject.toml/uv.lock)
cd <module> && uv sync

# modules that ship a requirements.txt
cd <module> && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# modules with no requirements file — install from the root requirements.txt
pip install -r requirements.txt        # run from repo root
```

### Run pattern by program type

| Type | How to run | Notes |
|------|-----------|-------|
| **Jupyter notebook** (`.ipynb`) | `uv run jupyter lab` / `jupyter lab`, or open in VS Code and pick the module's kernel | Needs `ipykernel`, `jupyter`. Run cells top-to-bottom. |
| **Plain Python script** (`.py`) | `python file.py` (or `uv run python file.py`) | Requires `.env` with the relevant API key (see table below). |
| **Streamlit app** | `streamlit run file.py` → http://localhost:8501 | Needs `streamlit`. Add `--server.port 8502` to change port. |
| **Gradio app** | `python file.py` → http://localhost:7860 | Needs `gradio`. Some accept `--share`, `--port`, `--setup-db`. |
| **FastAPI / REST server** | `python server.py` **or** `uvicorn <module>:app --reload --port <p>` | Needs `fastapi`, `uvicorn`. |
| **MCP server** | `python <mcp_server>.py` (SSE/HTTP) or launched by its client (stdio) | Needs `fastmcp`/`mcp`. |
| **Discord bot** | `python bot.py` | Needs `nextcord`, a `config.json`, and `DISCORD_TOKEN`. |
| **CrewAI / LangGraph agent** | `python file.py` | Needs `crewai`/`langgraph` + `OPENROUTER_API_KEY`. |

> Prefix any of these with `uv run` inside a uv-managed module to skip manual activation
> (e.g. `uv run streamlit run app.py`).

### BaseCamp1 — Python fundamentals
Deps: `uv sync` (uses `pyproject.toml`). Key packages: `groq`, `pandas`, `numpy`, `python-dotenv`, `ipykernel`.

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/1_My_First_Prog.py` | `uv run python Day_1/1_My_First_Prog.py` | — |
| `Day_1/*.ipynb` (Jupyter, functions, OOP, file I/O) | `uv run jupyter lab` then open the notebook | — |
| `Day_1/7_Groq.ipynb` | open in Jupyter | `GROQ_API_KEY` |
| `Day_2/*.ipynb` (lists…threading) | `uv run jupyter lab` | — |
| `Day_2/12_Process.py` | `uv run python Day_2/12_Process.py` | — |

### BaseCamp2 — OOP & web dev
Deps: `pip install -r Day_1/requirements.txt` (or `Day_2/`). Key packages: `fastapi`, `uvicorn`, `streamlit`, `pydantic`.

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/2a_calculator.py` … `6c_products.py` (FastAPI services) | `uvicorn 2a_calculator:app --reload --port 8000` (run from `Day_1/`) | `fastapi`, `uvicorn` |
| `Day_2/01_streamlit.py` … `09_streamlit.py` | `streamlit run Day_2/0N_streamlit.py` | `streamlit` |

### Week 1 — Pandas, NLP & Search
Deps: `uv sync` (pyproject) or `pip install -r Week1/requirements.txt`. Key packages: `pandas`, `pandasql`, `streamlit`, `nltk`, `PyPDF2`, `matplotlib`.

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/*.ipynb` (pandas, nlp, web_crawl, generate) | `uv run jupyter lab` | — |
| `Day_1/search_engine.py` | `python Day_1/search_engine.py` | reads `bigram/trigram/unigram_probs.json` |
| `Day_1/search_ui.py` | `streamlit run Day_1/search_ui.py` | `streamlit` |

### Week 2 — ML models & GenAI
Deps: `pip install -r Week2/requirements.txt`. Key packages: `transformers`, `torch`, `scikit-learn`, `streamlit`. **Requires `HF_TOKEN`.**

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/*.ipynb`, `Day_2/*.ipynb` | `jupyter lab` | `HF_TOKEN` |
| `Day_2/chatbot.py` | `streamlit run Day_2/chatbot.py` | `HF_TOKEN`, `streamlit` |
| `Day_2/search_engine.py` / `search_ui.py` | `python search_engine.py` / `streamlit run search_ui.py` | — |

### Week 3 — Coding assistant
Deps: `pip install -r Week3/requirements.txt`. Key packages: `groq`. **Requires `GROQ_API_KEY`.**

| Program | Command | Requires |
|---------|---------|----------|
| `coding_assistant.ipynb` | `jupyter lab` | `GROQ_API_KEY` |
| `coding_assistant.py` | `streamlit run coding_assistant.py` | `GROQ_API_KEY`, `streamlit` |

### Week 4 — RAG intro
Deps: root `requirements.txt`. Key packages: `langchain-community`, `qdrant-client`, `sentence-transformers`. **Requires `GOOGLE_API_KEY`.**

| Program | Command | Requires |
|---------|---------|----------|
| `Rag.ipynb` | `jupyter lab` | `GOOGLE_API_KEY` |

### Week 5 — Discord bot & advanced RAG
Deps: `pip install -r Week5/bot/requirements.txt`. Key packages: `nextcord`, `langchain`, `qdrant-client`.

| Program | Command | Requires |
|---------|---------|----------|
| `bot/bot.py` | `python bot/bot.py` (from `Week5/`) | `DISCORD_TOKEN`, `bot/config.json` (copy `sample_config.json`) |
| `ChatAssistant.ipynb`, `RagAdvanced.ipynb` | `jupyter lab` | `GOOGLE_API_KEY` |

### Week 6 — Data integration & function calling
Deps: root `requirements.txt` + `mysql-connector-python`. Key packages: `langchain`, `qdrant-client`, `rank_bm25`, `mysql-connector-python`. **Requires `GOOGLE_API_KEY`.**

| Program | Command | Requires |
|---------|---------|----------|
| `1_MySQL_Data.ipynb` … `11_UnStr_to_Str.ipynb` | `jupyter lab` | MySQL running for the MySQL notebooks |
| `5_Refined_Pipeline.py` | `python 5_Refined_Pipeline.py` | `GOOGLE_API_KEY` |

### Week 7 — Multimodal foundations
Deps: `pip install -r Week7/requirements.txt`. Key packages: `transformers`, `librosa`, `Pillow`, `nltk` (BLEU), `torch`.

| Program | Command | Requires |
|---------|---------|----------|
| `session_1/*.ipynb`, `session_2/*.ipynb` | `jupyter lab` | — |

### Week 8 — Embeddings, fusion & diffusion
Deps: `pip install -r Week8/requirements.txt`. Key packages: `sentence-transformers`, `transformers`, `diffusers`, `torch`, `Pillow`. **Requires `HF_TOKEN`.**

| Program | Command | Requires |
|---------|---------|----------|
| `session_1/*.ipynb` (text/image/audio embeddings, fusion) | `jupyter lab` | `HF_TOKEN` |
| `session_2/*.ipynb` (clip, blip, diffusion) | `jupyter lab` (GPU recommended) | `HF_TOKEN` |
| `assignment/*.ipynb` | `jupyter lab` | `HF_TOKEN` |

### Week 9 — Vector DBs & production RAG
Deps: `pip install -r Week9/requirements.txt`. Key packages: `lancedb`, `gradio`, `transformers` (CLIP/Qwen), `torch`.

| Program | Command | Requires |
|---------|---------|----------|
| `session_1/introduction_to_gradio.ipynb`, `lancedb_*.ipynb` | `jupyter lab` | — |
| `session_1/food_app.py`, `product_cataloger_app.py` | `python session_1/food_app.py` → :7860 | `gradio` |
| `session_2/app.py` (full RAG UI) | `python session_2/app.py` (`--setup-db` first run, `--share`, `--port 8080`) | `gradio`; `OPENAI_API_KEY` optional |
| `assignment/`, `solution/` fashion RAG | `python solution/solution_fashion_rag.py` | `gradio`, `lancedb` |

### Week 10 — Chatbots, REST & MCP
Deps: `pip install -r Week10/Day1/requirements.txt`. Key packages: `streamlit`, `fastapi`, `uvicorn`, `fastmcp`, `openai`. **Requires `OPENROUTER_API_KEY`.**

| Program | Command | Requires |
|---------|---------|----------|
| `Day1/chatbot_app.py`, `quiz_app.py`, `agent_chat_app.py` | `streamlit run Day1/chatbot_app.py` | `OPENROUTER_API_KEY` |
| `Day1/scripts/seed_db.py` | `python Day1/scripts/seed_db.py` | — |
| `Day2/1_rest_calculator_server.py` | `python Day2/1_rest_calculator_server.py` → :9321 | `fastapi`, `uvicorn` |
| `Day2/2_mcp_calculator_stdio_server.py` | started by its client notebook (stdio transport) | `fastmcp` |
| `Day2/3_mcp_calculator_sse_server.py` | `python Day2/3_mcp_calculator_sse_server.py` → SSE :9321 | `fastmcp` |
| `Day2/4_mcp_calculator_resource_server.py` | `python Day2/4_mcp_calculator_resource_server.py` | `fastmcp` |
| `Day2/*_client.ipynb` | `jupyter lab` (start the matching server first) | — |
| `Day2/5_streamlit_app.py` | `streamlit run Day2/5_streamlit_app.py` | `streamlit` |

### Week 11 — CrewAI agents
Deps: per-app requirements + root. Key packages: `crewai`, `crewai_tools`, `faiss-cpu`, `streamlit`. **Requires `OPENROUTER_API_KEY`** (+ `NEWS_API_KEY` for the aggregator).

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/*.ipynb` (agents, tasks, tools, workflows) | `jupyter lab` | `OPENROUTER_API_KEY` |
| `Day_2/1_annual_reports_analysis/main.py` | `python main.py` (from that folder) | `OPENROUTER_API_KEY`, FAISS index |
| `Day_2/2_news_aggregator/news_app.py` | `streamlit run news_app.py` | `NEWS_API_KEY`, `OPENROUTER_API_KEY` |
| `Day_2/3_sdlc_plan/sdlc_crew.py` | `python sdlc_crew.py` | `OPENROUTER_API_KEY` |
| `Day_2/4_code_assist/assist.py` | `streamlit run assist.py` | `OPENROUTER_API_KEY` |

### Week 12 — Advanced agents & orchestration
Deps: `pip install -r Week12/Day_2/3_financial_modelling/requirements.txt` (+ root). Key packages: `crewai`, `phidata`, `langgraph`, `pydantic`, `fastapi`.

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/1_structured_output.py`, `3a_csv_example.py`, `6_python_agent.py`, `7_plantUML.py` | `python Day_1/<file>.py` | `OPENROUTER_API_KEY` |
| `Day_1/4a_knowledge_Graph.ipynb` | `jupyter lab` | `OPENROUTER_API_KEY` |
| `Day_2/1_agent2agent/crewai_server.py` / `phidata_Server.py` | `python <server>.py`, then run the matching `a2a_client_{mac,windows}.py` | `OPENROUTER_API_KEY` |
| `Day_2/2_app_builder/coder.py` | `python coder.py` | `OPENROUTER_API_KEY` |
| `Day_2/3_financial_modelling/main.py` | `python main.py` (async orchestrator) | `OPENROUTER_API_KEY` |

### Week 13 — LangGraph workflows
Deps: `pip install -r "Week13/Day 2/requirements.txt"`. Key packages: `langgraph`, `langchain`, `streamlit`. **Requires `OPENROUTER_API_KEY`.**

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/simple_graph.py` | `python Day_1/simple_graph.py` | `OPENROUTER_API_KEY` |
| `Day_1/storygen.py`, `supporticket.py` | `streamlit run Day_1/storygen.py` | `OPENROUTER_API_KEY`, `streamlit` |
| `Day 2/codegen.py` | `streamlit run "Day 2/codegen.py"` | `OPENROUTER_API_KEY` |
| `Day 2/flight.py` | `python "Day 2/flight.py"` | `OPENROUTER_API_KEY` |
| `Day 2/Blog Writer Team/ui-agent.py` | `streamlit run "Day 2/Blog Writer Team/ui-agent.py"` | `OPENROUTER_API_KEY` |

### Week 14 — Advanced patterns
Deps: `pip install -r Week14/Day_1/requirements.txt` (or `Day2/`). Key packages: `langgraph`, `aiohttp`, `mcp`, `streamlit`. **Requires `OPENROUTER_API_KEY`.**

| Program | Command | Requires |
|---------|---------|----------|
| `Day_1/codereview.py`, `csv_1.py` | `streamlit run Day_1/codereview.py` | `OPENROUTER_API_KEY` |
| `Day2/arag.py` (async RAG) | `streamlit run Day2/arag.py` | `OPENROUTER_API_KEY` |
| `Day2/mcp.py` | `streamlit run Day2/mcp.py` | `OPENROUTER_API_KEY` |

### Week 15 — Batch & synthetic data
Deps: root `requirements.txt` + `anthropic`, `scipy`, `diffusers`. Key packages: `anthropic`, `scipy`, `diffusers`, `crewai`.

| Program | Command | Requires |
|---------|---------|----------|
| `Day1/batch.ipynb`, `image.ipynb`, `tokens.ipynb`, `predicted_output.ipynb` | `jupyter lab` | Claude API key |
| `Day2/2_…`–`7_crew_text_synthetic_data.ipynb` | `jupyter lab` | `OPENROUTER_API_KEY` for the CrewAI notebook |

### BuildWeek — Fine-tuning
`.excalidraw` files are whiteboards (open in Excalidraw / the VS Code Excalidraw extension), **not** runnable code. `shakespeare.txt` is a corpus consumed by fine-tuning notebooks.

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
- Notebooks under `BaseCamp1/Day_1/` are actively edited during class — expect frequent uncommitted changes there.
- `Git Command Reference` above covers the common workflow; see `commands.md` for the full guide plus FastAPI/Streamlit run/stop commands.
