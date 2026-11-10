import {
  BenchmarkRecord,
  CostCalculationRequest,
  CostCalculationResponse,
  DatasetRecord,
  EvaluationRecord,
  ExperimentRecord,
  GenerateResponse,
  HardwareProfile,
  KVCacheStats,
  ModelRecord,
  QuantizationRecord,
  TrainingJob,
} from '../types';

const API_BASE = '/api';

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });
    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(`API Error [${res.status}]: ${errBody || res.statusText}`);
    }
    return await res.json();
  } catch (error) {
    console.error(`Request to ${url} failed:`, error);
    throw error;
  }
}

export const api = {
  // System & Hardware
  getHealth: () => request<{ status: string; service: string; version: string }>('/health'),
  getHardware: () => request<HardwareProfile>('/hardware'),

  // Models
  getModels: (category?: string) =>
    request<ModelRecord[]>(`/models${category ? `?category=${category}` : ''}`),
  getModel: (id: string) => request<ModelRecord>(`/models/${encodeURIComponent(id)}`),

  // Datasets
  getDatasets: () => request<DatasetRecord[]>('/datasets'),
  getDatasetStats: (name = 'billsum') =>
    request<Record<string, any>>(`/datasets/${name}/stats`),

  // Evaluations
  getEvaluations: (modelId?: string) =>
    request<EvaluationRecord[]>(`/evaluation/results${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''}`),
  runEvaluation: (payload: {
    model_id: string;
    dataset_name?: string;
    split?: string;
    num_samples?: number;
    compare_with_api?: string;
  }) =>
    request<Record<string, any>>('/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Experiments Registry
  getExperiments: (category?: string) =>
    request<ExperimentRecord[]>(`/experiments${category ? `?category=${category}` : ''}`),

  // Fine-Tuning
  getTrainingJobs: () => request<TrainingJob[]>('/training/jobs'),
  startTraining: (payload: {
    base_model_id: string;
    method?: string;
    r?: number;
    lora_alpha?: number;
    lora_dropout?: number;
    num_train_epochs?: number;
    learning_rate?: number;
  }) =>
    request<Record<string, any>>('/training/train', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Quantization
  getQuantizations: () => request<QuantizationRecord[]>('/quantization/results'),
  getPrecisionMatrix: (modelId = 'Qwen/Qwen2.5-0.5B-Instruct') =>
    request<any[]>(`/quantization/compare-precisions?base_model_id=${encodeURIComponent(modelId)}`),
  runQuantization: (payload: {
    base_model_id: string;
    method: string;
    target_precision: string;
  }) =>
    request<Record<string, any>>('/quantization/quantize', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Inference & Benchmarking
  generateText: (payload: {
    model_id: string;
    prompt: string;
    max_output_tokens?: number;
    temperature?: number;
    top_p?: number;
  }) =>
    request<GenerateResponse>('/inference/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getKVCacheStats: (batchSize = 4, contextLength = 2048, precision = 'fp16') =>
    request<KVCacheStats>(
      `/inference/kv-cache-stats?batch_size=${batchSize}&context_length=${contextLength}&precision=${precision}`
    ),
  getBenchmarks: (modelId?: string) =>
    request<BenchmarkRecord[]>(`/benchmark/results${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''}`),
  runLoadTest: (payload: {
    model_id: string;
    concurrency_levels?: number[];
    requests_per_level?: number;
    prompt?: string;
    max_output_tokens?: number;
  }) =>
    request<Record<string, any>>('/benchmark/load-test', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Cost Analysis & Frontier ROI Modeling
  calculateCost: (payload: CostCalculationRequest) =>
    request<CostCalculationResponse>('/cost/calculate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getFrontierPresets: () => request<Record<string, any>>('/cost/frontier-presets'),
  getHardwarePresets: () => request<Record<string, any>>('/cost/hardware-presets'),
  getParetoAnalysis: () => request<Record<string, any>>('/cost/pareto-analysis'),
};
