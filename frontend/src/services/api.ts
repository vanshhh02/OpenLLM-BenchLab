import {
  ApiKeyItem,
  AsyncJobItem,
  AuthMeResponse,
  AuthTokenResponse,
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
  WorkerHealthResponse,
  WorkspaceMemberItem,
  WorkspaceSummary,
} from '../types';

const API_BASE = '/api';
const ACCESS_TOKEN_KEY = 'benchlab_access_token';
const REFRESH_TOKEN_KEY = 'benchlab_refresh_token';
const WORKSPACE_ID_KEY = 'benchlab_workspace_id';

export const authStorage = {
  getAccessToken: () => localStorage.getItem(ACCESS_TOKEN_KEY),
  getRefreshToken: () => localStorage.getItem(REFRESH_TOKEN_KEY),
  getWorkspaceId: () => localStorage.getItem(WORKSPACE_ID_KEY) || 'ws_default',
  setSession: (auth: AuthTokenResponse) => {
    localStorage.setItem(ACCESS_TOKEN_KEY, auth.access_token);
    localStorage.setItem(REFRESH_TOKEN_KEY, auth.refresh_token);
    localStorage.setItem(WORKSPACE_ID_KEY, auth.workspace.id);
  },
  setWorkspaceId: (workspaceId: string) => {
    localStorage.setItem(WORKSPACE_ID_KEY, workspaceId);
  },
  clear: () => {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.setItem(WORKSPACE_ID_KEY, 'ws_default');
  },
};

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const token = authStorage.getAccessToken();
  const workspaceId = authStorage.getWorkspaceId();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Workspace-ID': workspaceId,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...((options?.headers as Record<string, string>) || {}),
  };

  try {
    const res = await fetch(url, {
      ...options,
      headers,
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

  // Phase 8: Authentication & API Keys
  register: async (payload: {
    email: string;
    password: string;
    full_name?: string;
    workspace_name?: string;
  }) => {
    const res = await request<AuthTokenResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    authStorage.setSession(res);
    return res;
  },
  login: async (payload: { email: string; password: string; workspace_id?: string }) => {
    const res = await request<AuthTokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    authStorage.setSession(res);
    return res;
  },
  getMe: () => request<AuthMeResponse>('/auth/me'),
  getApiKeys: () => request<ApiKeyItem[]>('/auth/api-keys'),
  createApiKey: (payload: { name: string; scopes?: string[]; expires_in_days?: number }) =>
    request<ApiKeyItem>('/auth/api-keys', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  revokeApiKey: (keyId: string) =>
    request<ApiKeyItem>(`/auth/api-keys/${encodeURIComponent(keyId)}`, {
      method: 'DELETE',
    }),

  // Phase 8: Workspaces & RBAC
  getWorkspaces: () => request<WorkspaceSummary[]>('/workspaces'),
  createWorkspace: (payload: { name: string; description?: string }) =>
    request<WorkspaceSummary>('/workspaces', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getWorkspaceMembers: (workspaceId: string) =>
    request<WorkspaceMemberItem[]>(`/workspaces/${encodeURIComponent(workspaceId)}/members`),
  addWorkspaceMember: (workspaceId: string, payload: { email: string; role: string }) =>
    request<WorkspaceMemberItem>(`/workspaces/${encodeURIComponent(workspaceId)}/members`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateWorkspaceMemberRole: (workspaceId: string, userId: string, role: string) =>
    request<WorkspaceMemberItem>(
      `/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
      {
        method: 'PATCH',
        body: JSON.stringify({ role }),
      }
    ),
  removeWorkspaceMember: (workspaceId: string, userId: string) =>
    request<{ status: string }>(
      `/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
      { method: 'DELETE' }
    ),

  // Phase 8: Async Job Queue & Workers
  getWorkerHealth: () => request<WorkerHealthResponse>('/jobs/worker/health'),
  getJobs: (status?: string, jobType?: string) => {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (jobType) params.set('job_type', jobType);
    const qs = params.toString();
    return request<AsyncJobItem[]>(`/jobs${qs ? `?${qs}` : ''}`);
  },
  submitJob: (payload: {
    job_type: 'training' | 'evaluation' | 'quantization' | 'load_test';
    payload: Record<string, any>;
  }) =>
    request<AsyncJobItem>('/jobs/submit', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  cancelJob: (jobId: string) =>
    request<AsyncJobItem>(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    }),
};

