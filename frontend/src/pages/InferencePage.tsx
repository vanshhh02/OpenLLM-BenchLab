import React, { useEffect, useState } from 'react';
import {
  Zap,
  Play,
  Activity,
  HardDrive,
  BarChart2,
  Clock,
  Cpu,
  Layers,
  Send,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';
import { GenerateResponse, KVCacheStats } from '../types';

export const InferencePage: React.FC = () => {
  // Playground state
  const [modelId, setModelId] = useState<string>('Qwen/Qwen2.5-0.5B-Instruct');
  const [prompt, setPrompt] = useState<string>(
    'Summarize this legislative clause: Section 1. Small Business Energy Efficiency Act. Directs the Small Business Administration to establish a grant program providing up to $50,000 for verified solar and renewable facility upgrades.'
  );
  const [maxTokens, setMaxTokens] = useState<number>(64);
  const [temperature, setTemperature] = useState<number>(0.1);
  const [generating, setGenerating] = useState<boolean>(false);
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);

  // KV Cache state
  const [batchSize, setBatchSize] = useState<number>(4);
  const [contextLength, setContextLength] = useState<number>(2048);
  const [precision, setPrecision] = useState<string>('fp16');
  const [kvStats, setKvStats] = useState<KVCacheStats | null>(null);

  // Load testing state
  const [benchRunning, setBenchRunning] = useState<boolean>(false);
  const [benchSummary, setBenchSummary] = useState<any>(null);

  // Fetch KV Cache stats
  useEffect(() => {
    async function loadKV() {
      try {
        const stats = await api.getKVCacheStats(batchSize, contextLength, precision);
        setKvStats(stats);
      } catch (e) {
        console.error('Failed to calculate KV cache stats:', e);
      }
    }
    loadKV();
  }, [batchSize, contextLength, precision]);

  const handleGenerate = async () => {
    setGenerating(true);
    setGenResult(null);
    try {
      const res = await api.generateText({
        model_id: modelId,
        prompt,
        max_output_tokens: maxTokens,
        temperature,
      });
      setGenResult(res);
    } catch (e) {
      console.error('Generation failed:', e);
      alert('Generation error. Ensure model weights are loaded.');
    } finally {
      setGenerating(false);
    }
  };

  const handleRunLoadTest = async () => {
    setBenchRunning(true);
    setBenchSummary(null);
    try {
      const res = await api.runLoadTest({
        model_id: modelId,
        concurrency_levels: [1, 2, 4],
        requests_per_level: 2,
        max_output_tokens: 16,
      });
      setBenchSummary(res);
    } catch (e) {
      console.error('Load test failed:', e);
      alert('Load test failed. Check server logs.');
    } finally {
      setBenchRunning(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-amber-400 uppercase tracking-wider">
          <Zap className="w-4 h-4" />
          Phase 5: Inference Serving, KV Cache Analysis & Load Testing
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
          Serving Playground & Continuous Batching
        </h1>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-2xl">
          Test real-time model inference, measure Time-To-First-Token (TTFT) and throughput,
          calculate KV Cache memory scaling across context windows, and execute multi-client
          concurrency stress tests.
        </p>
      </div>

      {/* Live Generation Playground */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-5">
        <h2 className="text-sm font-bold text-white flex items-center gap-2">
          <Play className="w-4 h-4 text-emerald-400" />
          Live Inference Playground
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 space-y-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Target Model</label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
              >
                <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen/Qwen2.5-0.5B-Instruct</option>
                <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen/Qwen2.5-1.5B-Instruct</option>
                <option value="google/gemma-2-2b-it">google/gemma-2-2b-it</option>
              </select>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Input Prompt (Bill Clause)</span>
                <button
                  type="button"
                  onClick={() =>
                    setPrompt(
                      'SEC. 3. RESEARCH AND DEVELOPMENT INITIATIVE.\nAuthorizes the National Science Foundation to allocate $250,000,000 across 5 fiscal years for advanced quantum computational infrastructure and university research grants.'
                    )
                  }
                  className="text-emerald-400 hover:underline text-[11px]"
                >
                  Load Sample Bill
                </button>
              </div>
              <textarea
                rows={4}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full p-3 rounded-xl bg-slate-800 border border-slate-700 text-xs text-slate-200 font-mono focus:border-emerald-500 focus:outline-none resize-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Max Tokens:</span>
                  <span className="font-mono text-white">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="16"
                  max="256"
                  step="16"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Temperature:</span>
                  <span className="font-mono text-white">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            <button
              disabled={generating}
              onClick={handleGenerate}
              className="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 transition-all"
            >
              {generating ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Generating Summary...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Generate Model Summary
                </>
              )}
            </button>
          </div>

          {/* Playground Output */}
          <div className="lg:col-span-5 flex flex-col justify-between p-4 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-4">
            <div>
              <span className="text-xs font-medium uppercase tracking-wider text-slate-400 block mb-2">
                Generated Response
              </span>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 min-h-[140px] text-xs text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">
                {genResult?.generated_text || (
                  <span className="text-slate-500 italic">
                    Output will appear here after clicking Generate Summary.
                  </span>
                )}
              </div>
            </div>

            {/* Serving Performance Telemetry */}
            {genResult && (
              <div className="pt-2 border-t border-slate-700/80">
                <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1.5">
                  Real-Time Serving Metrics
                </span>
                <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[9px]">TTFT</span>
                    <span className="text-emerald-400 font-bold">
                      {genResult.time_to_first_token_ms.toFixed(0)} ms
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[9px]">THROUGHPUT</span>
                    <span className="text-white font-bold">
                      {genResult.tokens_per_second.toFixed(1)} tps
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[9px]">TOTAL TIME</span>
                    <span className="text-slate-300">
                      {genResult.total_latency_ms.toFixed(0)} ms
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* KV Cache Scaling Visualizer */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-blue-400" />
              KV Cache Memory Allocation Estimator
            </h2>
            <p className="text-xs text-slate-400 font-mono">
              Memory = 2 × 2 × Layers × Heads × Head_Dim × Tokens × Batch × Bytes_Precision
            </p>
          </div>
          <div className="text-right">
            <span className="text-xs text-slate-400 block">Total KV Footprint:</span>
            <span className="text-xl font-bold font-mono text-blue-400">
              {kvStats ? `${kvStats.total_memory_mb.toFixed(2)} MB` : '--'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Batch Size:</span>
              <span className="font-mono text-white font-bold">{batchSize}</span>
            </div>
            <input
              type="range"
              min="1"
              max="32"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Context Tokens:</span>
              <span className="font-mono text-white font-bold">
                {contextLength.toLocaleString()}
              </span>
            </div>
            <input
              type="range"
              min="512"
              max="8192"
              step="512"
              value={contextLength}
              onChange={(e) => setContextLength(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Cache Precision</label>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white focus:border-blue-500 focus:outline-none"
            >
              <option value="fp16">FP16 / BF16 (2 Bytes/Elem)</option>
              <option value="fp32">FP32 (4 Bytes/Elem)</option>
              <option value="int8">INT8 Quantized (1 Byte/Elem)</option>
              <option value="int4">INT4 Quantized (0.5 Byte/Elem)</option>
            </select>
          </div>
        </div>

        {/* KV Cache Scaling Curve Grid */}
        <div className="mt-4 pt-3 border-t border-slate-800/80">
          <span className="text-[11px] text-slate-400 block mb-2 font-medium">
            Scaling Curve (Batch 1 to 16, Context 512 to 8,192):
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
            {kvStats?.scaling_curve.map((pt) => (
              <div
                key={`${pt.batch_size}-${pt.context_length}`}
                className={`p-2.5 rounded-lg text-center font-mono text-[11px] border ${
                  pt.batch_size === batchSize && pt.context_length === contextLength
                    ? 'bg-blue-500/20 border-blue-500/50 text-white font-bold'
                    : 'bg-slate-800/40 border-slate-700/40 text-slate-300'
                }`}
              >
                <div className="text-[9px] text-slate-500 uppercase">
                  B{pt.batch_size} • {pt.context_length}T
                </div>
                <div className="mt-1 text-blue-300">{pt.memory_mb.toFixed(1)} MB</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Concurrent Load Tester */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              Concurrent Load Testing & Latency Percentiles
            </h2>
            <p className="text-xs text-slate-400">
              Measure P50, P95, and P99 latency degradation as concurrency scales from 1 to 4 workers.
            </p>
          </div>
          <button
            disabled={benchRunning}
            onClick={handleRunLoadTest}
            className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-emerald-600/20 transition-all"
          >
            {benchRunning ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Executing Concurrent Sweep...
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                Run Concurrency Load Test
              </>
            )}
          </button>
        </div>

        {benchSummary && (
          <div className="overflow-x-auto mt-4">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-800/80 text-slate-400">
                <tr>
                  <th className="py-2.5 px-3 rounded-l-lg">Concurrency Level</th>
                  <th className="py-2.5 px-3">Reqs (Success/Fail)</th>
                  <th className="py-2.5 px-3">P50 Latency</th>
                  <th className="py-2.5 px-3">P95 Latency</th>
                  <th className="py-2.5 px-3">P99 Latency</th>
                  <th className="py-2.5 px-3">Throughput (TPS)</th>
                  <th className="py-2.5 px-3 rounded-r-lg">KV VRAM</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {benchSummary.levels.map((lvl: any) => (
                  <tr key={lvl.concurrency_level} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-3 font-mono font-bold text-white">
                      {lvl.concurrency_level} Worker(s)
                    </td>
                    <td className="py-2.5 px-3 font-mono">
                      {lvl.successful_requests} / {lvl.failed_requests}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-emerald-400">
                      {lvl.p50_latency_ms.toFixed(0)} ms
                    </td>
                    <td className="py-2.5 px-3 font-mono text-amber-400">
                      {lvl.p95_latency_ms.toFixed(0)} ms
                    </td>
                    <td className="py-2.5 px-3 font-mono text-rose-400">
                      {lvl.p99_latency_ms.toFixed(0)} ms
                    </td>
                    <td className="py-2.5 px-3 font-mono font-bold text-white">
                      {lvl.tokens_per_second.toFixed(1)} tps
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-400">
                      {lvl.kv_cache_usage_mb.toFixed(1)} MB
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

