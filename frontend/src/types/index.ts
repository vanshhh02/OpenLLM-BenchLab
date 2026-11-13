export interface HardwareProfile {
  platform_system: string;
  platform_release: string;
  platform_machine: string;
  cpu_brand: string;
  cpu_count_physical: number;
  cpu_count_logical: number;
  total_ram_gb: number;
  available_ram_gb: number;
  accelerator_type: 'cuda' | 'mps' | 'cpu';
  accelerator_name: string;
  accelerator_count: number;
  accelerator_memory_gb?: number | null;
  bf16_supported: boolean;
  fp16_supported: boolean;
  int8_supported: boolean;
  int4_bitsandbytes_supported: boolean;
  vllm_supported: boolean;
  llamacpp_supported: boolean;
  notes: string[];
}

export interface ModelRecord {
  id: string;
  name: string;
  provider: string;
  architecture: string;
  parameter_count_b: number;
  context_window: number;
  precision: string;
  is_commercial_api: boolean;
  is_fine_tuned: boolean;
  is_quantized: boolean;
  base_model_id?: string | null;
  adapter_path?: string | null;
  quantization_method?: string | null;
  license: string;
  created_at?: string;
}

export interface DatasetRecord {
  id: string;
  name: string;
  description?: string;
  train_count: number;
  val_count: number;
  test_count: number;
  avg_input_tokens: number;
  avg_summary_tokens: number;
  max_input_tokens: number;
  max_summary_tokens: number;
  sample_records?: any[];
}

export interface EvaluationRecord {
  id: string;
  experiment_id?: string | null;
  model_id: string;
  dataset_name: string;
  split: string;
  num_samples: number;
  rouge1: number;
  rouge2: number;
  rougeL: number;
  bert_score?: number | null;
  factual_consistency?: number | null;
  avg_input_length: number;
  avg_output_length: number;
  avg_latency_ms: number;
  avg_ttft_ms: number;
  tokens_per_second: number;
  peak_memory_mb: number;
  cost_per_request_usd: number;
  created_at?: string;
}

export interface TrainingJob {
  id: string;
  base_model_id: string;
  method: 'lora' | 'qlora';
  status: 'pending' | 'running' | 'completed' | 'failed';
  train_loss?: number | null;
  val_loss?: number | null;
  trainable_parameters?: number | null;
  total_parameters?: number | null;
  trainable_percentage?: number | null;
  duration_seconds?: number | null;
  checkpoint_path?: string | null;
  checkpoint_size_mb?: number | null;
  peak_memory_mb?: number | null;
  error_message?: string | null;
  created_at?: string;
  completed_at?: string | null;
}

export interface QuantizationRecord {
  id: string;
  base_model_id: string;
  quantized_model_id: string;
  method: string;
  target_precision: string;
  original_memory_mb: number;
  quantized_memory_mb: number;
  memory_reduction_pct: number;
  baseline_rougel?: number | null;
  quantized_rougel?: number | null;
  rougel_delta?: number | null;
  baseline_latency_ms?: number | null;
  quantized_latency_ms?: number | null;
  speedup_ratio?: number | null;
  status: string;
  artifact_path?: string | null;
  notes?: string | null;
}

export interface BenchmarkRecord {
  id: string;
  model_id: string;
  benchmark_type: string;
  concurrency_level: number;
  num_requests: number;
  successful_requests: number;
  failed_requests: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  avg_latency_ms: number;
  avg_ttft_ms: number;
  requests_per_second: number;
  tokens_per_second: number;
  peak_memory_mb: number;
  engine: string;
  kv_cache_usage_mb?: number | null;
  created_at?: string;
}

export interface ExperimentRecord {
  id: string;
  name: string;
  model_id: string;
  category: string;
  rouge1?: number | null;
  rouge2?: number | null;
  rougeL?: number | null;
  bert_score?: number | null;
  avg_latency_ms?: number | null;
  tokens_per_second?: number | null;
  peak_memory_mb?: number | null;
  cost_per_1k_requests?: number | null;
  status: string;
  parameters?: Record<string, any>;
  metrics?: Record<string, any>;
  created_at?: string;
}

export interface CostCalculationRequest {
  monthly_requests: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  api_name: string;
  api_input_cost_per_m: number;
  api_output_cost_per_m: number;
  server_monthly_rental_usd: number;
  hardware_purchase_cost_usd: number;
  amortization_months: number;
  server_power_watts: number;
  electricity_kwh_cost_usd: number;
  operational_overhead_monthly_usd: number;
  tokens_per_second_local: number;
}

export interface SensitivityPoint {
  monthly_requests: number;
  api_monthly_cost_usd: number;
  local_monthly_cost_usd: number;
  savings_usd: number;
  savings_percentage: number;
  recommended_architecture: 'frontier_api' | 'self_hosted';
}

export interface CostCalculationResponse {
  monthly_requests: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  api_cost_per_request_usd: number;
  api_cost_per_1k_usd: number;
  monthly_api_cost_usd: number;
  local_fixed_cost_monthly_usd: number;
  local_electricity_cost_monthly_usd: number;
  local_variable_cost_monthly_usd: number;
  monthly_local_cost_usd: number;
  local_cost_per_request_usd: number;
  local_cost_per_1k_usd: number;
  break_even_requests: number | null;
  break_even_status: string;
  monthly_savings_usd: number;
  savings_percentage: number;
  sensitivity_curve: SensitivityPoint[];
}

export interface GenerateResponse {
  model_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  generated_text: string;
  time_to_first_token_ms: number;
  total_latency_ms: number;
  tokens_per_second: number;
  peak_memory_mb: number;
  cost_usd: number;
}

export interface KVCacheStats {
  model_id: string;
  batch_size: number;
  context_length: number;
  precision: string;
  bytes_per_token_total: number;
  total_memory_bytes: number;
  total_memory_mb: number;
  scaling_curve: Array<{
    batch_size: number;
    context_length: number;
    memory_mb: number;
  }>;
}

// Phase 8: Auth, Multi-User Workspaces, API Keys & Async Jobs
export interface UserSummary {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at?: string;
}

export interface WorkspaceSummary {
  id: string;
  name: string;
  slug: string;
  description?: string;
  owner_id?: string;
  role: string;
  member_count: number;
  created_at?: string;
}

export interface AuthTokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in_seconds: number;
  user: UserSummary;
  workspace: WorkspaceSummary;
}

export interface AuthMeResponse {
  authenticated: boolean;
  auth_method: string;
  user?: UserSummary | null;
  workspace_id: string;
  workspace_role: string;
  scopes: string[];
}

export interface ApiKeyItem {
  id: string;
  name: string;
  key_prefix: string;
  workspace_id: string;
  scopes: string[];
  is_active: boolean;
  last_used_at?: string;
  expires_at?: string;
  created_at?: string;
  raw_key?: string;
}

export interface WorkspaceMemberItem {
  id: string;
  workspace_id: string;
  user_id: string;
  email: string;
  full_name: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  created_at?: string;
}

export interface AsyncJobItem {
  id: string;
  workspace_id: string;
  created_by_user_id?: string;
  job_type: 'training' | 'evaluation' | 'quantization' | 'load_test';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  backend: string;
  celery_task_id?: string;
  payload_json: Record<string, any>;
  result_json?: Record<string, any> | null;
  progress_pct: number;
  current_step?: string;
  error_message?: string;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface WorkerHealthResponse {
  backend: string;
  celery_configured: boolean;
  active_local_futures: number;
  workspace_job_counts: Record<string, number>;
  total_jobs: number;
}
