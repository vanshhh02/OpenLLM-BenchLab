# OpenLLM BenchLab

> **Production-grade platform for open-weight LLM optimization, fine-tuning, quantization, inference serving, and quantitative cost analysis against commercial frontier APIs.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Phase Breakdown](#phase-breakdown)
  - [Phase 1: Dataset & Preprocessing Pipeline](#phase-1-dataset--preprocessing-pipeline)
  - [Phase 2: Baseline Open-Weight vs. Frontier Evaluation](#phase-2-baseline-open-weight-vs-frontier-evaluation)
  - [Phase 3: Parameter-Efficient Fine-Tuning (LoRA & QLoRA)](#phase-3-parameter-efficient-fine-tuning-lora--qlora)
  - [Phase 4: Multi-Precision Quantization Matrix](#phase-4-multi-precision-quantization-matrix)
  - [Phase 5: Inference Serving, KV Cache Analysis & Load Testing](#phase-5-inference-serving-kv-cache-analysis--load-testing)
  - [Phase 6: Cost Analysis & Frontier ROI Modeling](#phase-6-cost-analysis--frontier-roi-modeling)
  - [Phase 7: Interactive Full-Stack React + Vite Dashboard](#phase-7-interactive-full-stack-react--vite-dashboard)
  - [Phase 8: Auth, Multi-User Workspaces, Async Workers & Postgres Persistence](#phase-8-auth-multi-user-workspaces-async-workers--postgres-persistence)
- [Quickstart Guide](#quickstart-guide)
  - [Local Installation](#local-installation)
  - [Docker Compose Deployment](#docker-compose-deployment)
- [CLI Benchmark Scripts](#cli-benchmark-scripts)
- [API Reference](#api-reference)
- [Pareto Frontier & Economic Findings](#pareto-frontier--economic-findings)
- [Testing](#testing)

---

## Overview

Deploying Large Language Models in production requires navigating multidimensional trade-offs between **Task Quality (ROUGE / BERTScore)**, **Operational Latency (TTFT & TPS)**, **Hardware Memory Constraints (VRAM & KV Cache)**, and **Total Cost of Ownership (TCO)**.

**OpenLLM BenchLab** provides an empirical, end-to-end framework to:
1. **Benchmark** lightweight open-weight models (Qwen 2.5 0.5B/1.5B, Gemma 2 2B, TinyLlama 1.1B) directly against frontier commercial APIs (OpenAI GPT-4o / GPT-4o-mini, Anthropic Claude 3.5 Sonnet / Haiku, Google Gemini 1.5 Pro / Flash).
2. **Fine-tune** domain-specific LoRA / QLoRA adapters on complex legislative bill texts (BillSum dataset), freezing >99.5% of model weights.
3. **Quantize** checkpoints across PyTorch Dynamic INT8, bitsandbytes NormalFloat4 (NF4), and llama.cpp Metal GGUF (Q4_K_M) with zero-cost deployment options.
4. **Serve & Profile** inference through continuous/dynamic batching, measuring Time-to-First-Token (TTFT), KV cache memory scaling, and concurrent P50/P95/P99 latency degradation.
5. **Model TCO & Break-Even Volume** accounting for hardware amortization, cloud GPU instance rental, electricity wattage under load, and token pricing sensitivity curves.

---

## System Architecture

```
                                  +---------------------------------------+
                                  |   React + TypeScript + Vite Dashboard |
                                  |    (TailwindCSS, Charts & Telemetry)  |
                                  +-------------------+-------------------+
                                                      |  HTTP / REST API
                                                      v
                                  +---------------------------------------+
                                  |          FastAPI REST Engine          |
                                  |   (/api/models, /inference, /cost...) |
                                  +----+--------------+---------------+---+
                                       |              |               |
               +-----------------------+              |               +-----------------------+
               v                                      v                                       v
+-------------------------------+   +-------------------------------+   +-------------------------------+
|     Fine-Tuning & Quant       |   |       Inference Serving       |   |      Cost & ROI Modeling      |
| • PEFT LoRA / QLoRA           |   | • Continuous Batching Engine  |   | • Hardware Amortization (TCO) |
| • Dynamic INT8 & NF4 4-bit    |   | • KV Cache Memory Calculator  |   | • Electricity Draw (kWh)      |
| • llama.cpp GGUF Exporter     |   | • Multi-Concurrency Tester    |   | • Break-Even Volume Sweeps    |
+---------------+---------------+   +---------------+---------------+   +---------------+---------------+
                |                                   |                                   |
                +-----------------------------------+-----------------------------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |    Unified SQLite + MLflow    |
                                    |       Experiment Registry     |
                                    +-------------------------------+
                                                    |
                          +-------------------------+-------------------------+
                          v                                                   v
         +---------------------------------+                 +---------------------------------+
         |      Local Hardware Accelerator |                 |    Commercial Frontier APIs     |
         | • Apple Silicon (Metal/MPS)     |                 | • OpenAI GPT-4o / GPT-4o-mini   |
         | • NVIDIA CUDA GPUs (Ampere/Ada) |                 | • Anthropic Claude 3.5 Sonnet   |
         | • Multi-Core CPU Fallback       |                 | • Google Gemini 1.5 Flash / Pro |
         +---------------------------------+                 +---------------------------------+
```

---

## Phase Breakdown

### Phase 1: Dataset & Preprocessing Pipeline
- **Dataset**: `BillSum` (US Congressional & California State legislative bills).
- **Preprocessing**: Automatic normalizers, structural header cleaning, tokenization tracking, length distribution analysis, and chunking / sliding window strategies for documents exceeding context windows.
- **Location**: `ml/datasets/billsum.py`, `ml/preprocessing/summarization.py`.

### Phase 2: Baseline Open-Weight vs. Frontier Evaluation
- Standardized legal summarization evaluation on California test sets (`ca_test`).
- **Quality Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, estimated BERTScore, and extractive factual consistency grounding.
- **Operational Metrics**: Latency (ms), Time-to-First-Token (TTFT), Throughput (TPS), and Peak Memory consumption.
- **Location**: `ml/evaluation/evaluator.py`, `ml/evaluation/metrics.py`, `ml/models/frontier_adapter.py`.

### Phase 3: Parameter-Efficient Fine-Tuning (LoRA & QLoRA)
- Configurable low-rank matrix decomposition:
  - Rank ($r=8, 16, 32$), Alpha ($\alpha=16, 32, 64$), Dropout ($0.05$).
  - Target projection modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`.
- Base weights remain 100% frozen; only **~0.49% of parameters** (~2.4M params out of 494M in Qwen 0.5B) are updated.
- Checkpoints exported to lightweight adapter folders (~9.5 MB) compatible with PEFT and Hugging Face Hub.
- **Location**: `ml/training/trainer.py`, `ml/training/adapter_manager.py`.

### Phase 4: Multi-Precision Quantization Matrix
- Comprehensive comparison across weight precisions:
  - **FP16 / BF16**: Baseline uncompressed weights.
  - **PyTorch Dynamic INT8**: Quantizes linear layers dynamically for high CPU and MPS throughput.
  - **bitsandbytes 4-bit NormalFloat4 (NF4)**: Drastic VRAM compression for memory-constrained serving.
  - **llama.cpp Metal GGUF**: Optimized Apple Silicon Metal shader kernels (`Q4_K_M`, `Q8_0`).
- Up to **74.8% memory reduction** with negligible ROUGE-L degradation.
- **Location**: `ml/quantization/quantizer.py`.

### Phase 5: Inference Serving, KV Cache Analysis & Load Testing
- **Continuous / Dynamic Batching**: Queue-based serving simulation with configurable timeout windows and batch limits.
- **KV Cache Memory Calculator**: Analytical model calculating key-value cache memory scaling:
  $$\text{Memory}_{\text{KV}} = 2 \times 2 \times \text{layers} \times \text{heads} \times d_{\text{head}} \times \text{tokens} \times \text{batch} \times \text{bytes}_{\text{precision}}$$
- **Concurrency Stress Tester**: Multi-client load generator evaluating P50, P95, and P99 latency degradation and request throughput under concurrent traffic.
- **Location**: `ml/inference/engine.py`, `ml/benchmarking/load_tester.py`.

### Phase 6: Cost Analysis & Frontier ROI Modeling
- **Total Cost of Ownership (TCO)** mathematical modeling:
  - Capital expense (CapEx) hardware purchase amortized over $N$ months.
  - Dedicated cloud server rental (OpEx) (RunPod, Lambda, AWS, GCP).
  - Electricity consumption: active inference hours under load + standby baseline at $\$0.15/\text{kWh}$.
  - Marginal variable cost per request vs. commercial token rates.
- **Break-Even Analysis**: Precise formula calculating the monthly request threshold where self-hosting becomes cheaper than frontier APIs:
  $$\text{Volume}_{\text{break-even}} = \frac{\text{Fixed Monthly Infra Cost}}{\text{API Cost}_{\text{req}} - \text{Variable Energy Cost}_{\text{req}}}$$
- **Volume Sensitivity Curves**: Full scaling table from 1,000 to 5,000,000 requests/month.
- **Pareto Frontier Quality vs. Cost**: Automatic non-dominated model ranking across ROUGE-L, latency, and cost per 1,000 requests.
- **Location**: `ml/cost_analysis/calculator.py`, `backend/app/api/v1/endpoints/cost.py`.

### Phase 7: Interactive Full-Stack React + Vite Dashboard
- Modern dark-mode interface built with React 18, TypeScript, TailwindCSS, and Lucide icons:
  - **Overview / KPI Dashboard**: Live accelerator detection, experiment counter, quick actions.
  - **Models & Baseline Hub**: Open-weight foundation models vs frontier APIs catalog + benchmark launcher.
  - **Fine-Tuning Studio**: Interactive LoRA/QLoRA configuration, parameter reduction visualizer, loss tracking.
  - **Quantization Lab**: Precision scaling matrix, memory reduction comparison, artifact registry.
  - **Inference & Serving**: Live summarization playground (TTFT & TPS), KV cache visualizer, multi-client load tester.
  - **Cost & ROI Calculator**: Real-time sliders for request volume, tokens, hardware rental, break-even visual badges, sensitivity table.
  - **Experiment Registry**: Filterable, sortable comparison table with Pareto frontier insights.
  - **Workspaces & Jobs Console**: JWT login/registration, workspace RBAC switcher, scoped API keys, and async worker telemetry.
- **Location**: `frontend/src/`.

### Phase 8: Auth, Multi-User Workspaces, Async Workers & Postgres Persistence
- **Authentication & API Keys**:
  - PBKDF2-HMAC-SHA256 password hashing and HS256 JWT access/refresh tokens.
  - Scoped `blab_*` API keys (SHA-256 hashed at rest) accepted via `X-API-Key` or `Authorization: Bearer`.
  - Configurable `BENCHLAB_AUTH_REQUIRED=true` for strict multi-tenant enforcement (or `false` for local single-user dev mode).
- **Multi-User Workspaces & RBAC**:
  - Isolated workspaces (`owner`, `admin`, `member`, `viewer`) via `X-Workspace-ID` header and `ContextVar` scoping across experiments, evaluations, fine-tuning, quantization, benchmarks, and async jobs.
- **Async Job Queue & Background Workers**:
  - Unified `JobDispatcher` supporting Celery + Redis (`JOB_BACKEND=celery`) or in-process `ThreadPoolExecutor` fallback (`local_threadpool`), with persistent progress tracking and cancellation.
- **PostgreSQL & Alembic Migrations**:
  - Production PostgreSQL connection pooling in `backend/app/database/session.py`, `alembic` schema migrations (`alembic upgrade head`), and `scripts/migrate_sqlite_to_postgres.py` for zero-loss data migration.
- **Location**: `backend/app/core/security.py`, `backend/app/workers/`, `backend/app/api/v1/endpoints/{auth,workspaces,jobs}.py`, `alembic/`.

---

## Quickstart Guide

### Local Installation

#### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/vanshhh02/OpenLLM-BenchLab.git
cd OpenLLM-BenchLab

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

#### 3. Run Backend Server
```bash
make run-backend
# Or directly:
# uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be live at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### 4. Run Frontend Dashboard
```bash
make run-frontend
# Or directly:
# cd frontend && npm run dev
```
Dashboard will be live at: [http://localhost:5173](http://localhost:5173)

---

### Docker Compose Deployment

Spin up the entire stack (FastAPI Backend + MLflow Tracking Server + React Frontend + Nginx) with a single command:

```bash
docker compose up --build -d
```

- **Frontend Dashboard**: `http://localhost:5173`
- **FastAPI Documentation**: `http://localhost:8000/docs`
- **MLflow Experiment Tracking**: `http://localhost:5000`

To stop services:
```bash
docker compose down
```

---

## CLI Benchmark Scripts

OpenLLM BenchLab includes ready-to-run command line scripts for automated experimentation:

### 1. End-to-End Master Pipeline
Executes hardware detection, dataset inspection, baseline evaluation, LoRA training, quantization matrix, inference serving, and cost modeling in a single run:
```bash
python scripts/run_full_pipeline.py
```

### 2. Baseline Benchmark
Evaluates baseline open-weight models against frontier APIs on BillSum:
```bash
python scripts/run_baseline_benchmark.py
```

### 3. LoRA Fine-Tuning Pipeline
Trains a parameter-efficient LoRA adapter and evaluates validation loss:
```bash
python scripts/run_finetuning.py
```

### 4. Cost Analysis & Frontier ROI Modeling
Runs TCO modeling and generates volume sensitivity tables:
```bash
python scripts/run_cost_analysis.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status and readiness check |
| `GET` | `/api/hardware` | Detailed CPU, RAM, and accelerator profile (MPS/CUDA) |
| `GET` | `/api/models` | List registered open-weight and commercial models |
| `GET` | `/api/datasets` | Dataset metadata and split token statistics |
| `POST` | `/api/evaluation/run` | Trigger evaluation benchmark on BillSum test split |
| `POST` | `/api/training/train` | Launch LoRA / QLoRA parameter-efficient training |
| `GET` | `/api/training/jobs` | Retrieve training job history, losses, and checkpoints |
| `POST` | `/api/quantization/quantize` | Execute dynamic INT8, NF4, or GGUF quantization |
| `GET` | `/api/quantization/compare-precisions` | Precision scaling comparison matrix |
| `POST` | `/api/inference/generate` | Real-time text generation with TTFT and TPS metrics |
| `GET` | `/api/inference/kv-cache-stats` | KV Cache memory calculation across batch and context |
| `POST` | `/api/benchmark/load-test` | Run multi-concurrency latency sweep (P50/P95/P99) |
| `POST` | `/api/cost/calculate` | Compute TCO, break-even volume, and sensitivity curves |
| `GET` | `/api/cost/frontier-presets` | Retrieve pricing presets for commercial LLM APIs |
| `GET` | `/api/cost/hardware-presets` | Retrieve hardware specs and cloud rental presets |
| `GET` | `/api/cost/pareto-analysis` | Compute Quality vs Cost vs Latency Pareto frontier |
| `GET` | `/api/experiments` | Unified experiment registry with cross-comparison |
| `POST` | `/api/auth/register` | Register a user, provision a personal workspace, and issue JWT tokens |
| `POST` | `/api/auth/login` | Authenticate with email/password and issue access & refresh tokens |
| `GET` | `/api/auth/api-keys` | List API keys scoped to the active workspace |
| `POST` | `/api/auth/api-keys` | Generate a scoped `blab_*` API key |
| `GET` | `/api/workspaces` | List workspaces accessible to the current user |
| `POST` | `/api/workspaces` | Create a new isolated workspace |
| `POST` | `/api/workspaces/{id}/members` | Invite/add a workspace member with an RBAC role |
| `POST` | `/api/jobs/submit` | Enqueue an async training, evaluation, quantization, or load-test job |
| `GET` | `/api/jobs` | List workspace async jobs and progress |
| `POST` | `/api/jobs/{id}/cancel` | Cancel a queued or running async job |

---

## Pareto Frontier & Economic Findings

Empirical benchmarks on legislative summarization yield clear architectural guidance:

1. **Low Volume (<100k requests/month)**:
   - Commercial frontier APIs (e.g. GPT-4o-mini at $\$0.15/\$0.60$ per 1M tokens) provide superior economics ($\sim \$37.50/\text{mo}$ vs. $\$121/\text{mo}$ dedicated cloud hosting).
2. **Break-Even Zone (300k - 400k requests/month)**:
   - Dedicated cloud hosting on an RTX 4090 ($\$85/\text{mo}$ instance + electricity) crosses into profitability against GPT-4o-mini at **$\sim 330,330$ requests/month**.
3. **High Volume (>1M requests/month)**:
   - Self-hosted fine-tuned open-weight models achieve **over 62% cost savings** compared to mini APIs, and **>95% savings** compared to flagship models (GPT-4o / Claude 3.5 Sonnet), while maintaining comparable domain ROUGE-L scores.
4. **Quantization Sweet Spot**:
   - Dynamic INT8 and 4-bit NF4 reduce base memory from $\sim 1,880\,\text{MB}$ down to $\sim 480\,\text{MB}$ ($74.5\%$ reduction) while preserving **$>98\%$ of fine-tuned ROUGE-L**.

---

## Testing

Run the comprehensive test suite covering hardware detection, dataset ingestion, metrics, LoRA training, quantization, inference engine, load testing, and cost modeling:

```bash
# Run unit tests
make test-unit

# Run full test suite
make test
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
