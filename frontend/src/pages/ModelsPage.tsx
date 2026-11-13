import React, { useEffect, useState } from 'react';
import {
  Server,
  Play,
  Layers,
  Cpu,
  Sparkles,
  CheckCircle2,
  Clock,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { api } from '../services/api';
import { ModelRecord } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [evalModalOpen, setEvalModalOpen] = useState<boolean>(false);
  const [evalModelId, setEvalModelId] = useState<string>('Qwen/Qwen2.5-0.5B-Instruct');
  const [numSamples, setNumSamples] = useState<number>(5);
  const [compareApi, setCompareApi] = useState<string>('openai');
  const [runningEval, setRunningEval] = useState<boolean>(false);
  const [evalResult, setEvalResult] = useState<any>(null);

  useEffect(() => {
    async function loadModels() {
      try {
        const res = await api.getModels(selectedCategory === 'all' ? undefined : selectedCategory);
        setModels(res);
      } catch (e) {
        console.error('Failed to load models:', e);
      }
    }
    loadModels();
  }, [selectedCategory]);

  const handleRunEvaluation = async () => {
    setRunningEval(true);
    setEvalResult(null);
    try {
      const res = await api.runEvaluation({
        model_id: evalModelId,
        dataset_name: 'billsum',
        split: 'ca_test',
        num_samples: numSamples,
        compare_with_api: compareApi,
      });
      setEvalResult(res);
    } catch (e) {
      console.error('Evaluation run failed:', e);
      alert('Evaluation failed. Check server logs.');
    } finally {
      setRunningEval(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-blue-400 uppercase tracking-wider">
            <Server className="w-4 h-4" />
            Phase 1 & 2: Dataset, Models & Baseline Evaluation
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Model Hub & Baseline Benchmarks
          </h1>
          <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-2xl">
            Manage open-weight foundation models, fine-tuned adapters, quantized checkpoints,
            and commercial frontier APIs. Run standardized legal summarization evaluations.
          </p>
        </div>

        <button
          onClick={() => setEvalModalOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-lg shadow-emerald-600/20 transition-all"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          Run Baseline Benchmark
        </button>
      </div>

      {/* Category Tabs */}
      <div className="flex space-x-2 border-b border-slate-800 pb-3">
        {[
          { key: 'all', label: 'All Models' },
          { key: 'open_weight', label: 'Open-Weight' },
          { key: 'commercial_api', label: 'Frontier APIs' },
          { key: 'fine_tuned', label: 'Fine-Tuned Adapters' },
          { key: 'quantized', label: 'Quantized' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setSelectedCategory(tab.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              selectedCategory === tab.key
                ? 'bg-slate-800 text-white border border-slate-700'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {models.map((m) => (
          <div
            key={m.id}
            className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between">
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                    m.is_commercial_api
                      ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                      : m.is_fine_tuned
                      ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                      : m.is_quantized
                      ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                      : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  }`}
                >
                  {m.is_commercial_api
                    ? 'Frontier API'
                    : m.is_fine_tuned
                    ? 'LoRA Adapter'
                    : m.is_quantized
                    ? 'Quantized'
                    : 'Open-Weight Base'}
                </span>
                <span className="text-xs text-slate-500">{m.license}</span>
              </div>

              <h2 className="text-base font-bold text-white mt-3 truncate">{m.name}</h2>
              <div className="text-xs text-slate-400 font-mono mt-0.5">{m.id}</div>

              <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-300">
                <div className="p-2 rounded bg-slate-800/60">
                  <span className="text-[10px] text-slate-500 block">PARAMETERS</span>
                  <span className="font-mono font-medium">
                    {m.parameter_count_b > 0 ? `${m.parameter_count_b}B` : 'Proprietary'}
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-800/60">
                  <span className="text-[10px] text-slate-500 block">CONTEXT</span>
                  <span className="font-mono font-medium">
                    {m.context_window.toLocaleString()} tok
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-800/60">
                  <span className="text-[10px] text-slate-500 block">PRECISION</span>
                  <span className="font-mono font-medium uppercase">{m.precision}</span>
                </div>
                <div className="p-2 rounded bg-slate-800/60">
                  <span className="text-[10px] text-slate-500 block">PROVIDER</span>
                  <span className="capitalize">{m.provider}</span>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-3 border-t border-slate-800 flex items-center justify-between text-xs">
              <span className="text-slate-400">{m.architecture}</span>
              <button
                onClick={() => {
                  setEvalModelId(m.id);
                  setEvalModalOpen(true);
                }}
                className="text-emerald-400 hover:text-emerald-300 font-semibold flex items-center gap-1"
              >
                Benchmark <Play className="w-3 h-3" />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Benchmark Modal */}
      {evalModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl bg-slate-900 border border-slate-700 p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Play className="w-4 h-4 text-emerald-400" />
                Run Baseline Benchmark
              </h3>
              <button
                onClick={() => setEvalModalOpen(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Target Model</label>
                <select
                  value={evalModelId}
                  onChange={(e) => setEvalModelId(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white font-mono focus:border-emerald-500 focus:outline-none"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.id})
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Evaluation Samples</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={numSamples}
                    onChange={(e) => setNumSamples(Number(e.target.value))}
                    className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-white font-mono focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Compare With API</label>
                  <select
                    value={compareApi}
                    onChange={(e) => setCompareApi(e.target.value)}
                    className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-white focus:border-emerald-500 focus:outline-none"
                  >
                    <option value="openai">OpenAI GPT-4o-mini</option>
                    <option value="gemini">Google Gemini 1.5 Flash</option>
                    <option value="anthropic">Anthropic Claude 3.5 Haiku</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Run Button & Spinner */}
            <div className="pt-2 flex justify-end gap-2">
              <button
                onClick={() => setEvalModalOpen(false)}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium"
              >
                Cancel
              </button>
              <button
                disabled={runningEval}
                onClick={handleRunEvaluation}
                className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-emerald-600/20"
              >
                {runningEval ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running Inference & Scoring...
                  </>
                ) : (
                  <>Execute Benchmark</>
                )}
              </button>
            </div>

            {/* Live Result Callout */}
            {evalResult && (
              <div className="mt-4 p-4 rounded-xl bg-slate-800/80 border border-slate-700 text-xs space-y-2">
                <div className="font-bold text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" /> Benchmark Complete
                </div>
                <div className="grid grid-cols-3 gap-2 font-mono text-[11px] pt-1">
                  <div>
                    <span className="text-slate-500 block">ROUGE-1</span>
                    <span>{evalResult.quality?.rouge1.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">ROUGE-2</span>
                    <span>{evalResult.quality?.rouge2.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">ROUGE-L</span>
                    <span className="text-emerald-400 font-bold">
                      {evalResult.quality?.rougeL.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">LATENCY</span>
                    <span>{evalResult.operational?.avg_latency_ms.toFixed(0)} ms</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">TTFT</span>
                    <span>{evalResult.operational?.avg_ttft_ms.toFixed(0)} ms</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">THROUGHPUT</span>
                    <span>{evalResult.operational?.tokens_per_second.toFixed(1)} tps</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

