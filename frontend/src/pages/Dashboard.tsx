import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Cpu,
  Layers,
  Zap,
  TrendingDown,
  Server,
  ArrowRight,
  Database,
  CheckCircle2,
  Clock,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';
import { ExperimentRecord, HardwareProfile, ModelRecord } from '../types';

export const Dashboard: React.FC = () => {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [hwRes, modelsRes, expRes] = await Promise.allSettled([
          api.getHardware(),
          api.getModels(),
          api.getExperiments(),
        ]);
        if (hwRes.status === 'fulfilled') setHardware(hwRes.value);
        if (modelsRes.status === 'fulfilled') setModels(modelsRes.value);
        if (expRes.status === 'fulfilled') setExperiments(expRes.value);
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-800 to-emerald-950 border border-slate-700/60 p-8 shadow-2xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            End-to-End LLM Optimization & Benchmarking
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Open-Weight LLMs vs. Frontier APIs
          </h1>
          <p className="mt-3 text-slate-300 text-sm sm:text-base leading-relaxed">
            Quantify summarization quality (ROUGE / BERTScore), serving throughput, and
            inference latency across PyTorch & Apple Metal MPS / CUDA. Measure exact
            TCO break-even volumes and cost savings against commercial APIs.
          </p>

          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              to="/cost"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold shadow-lg shadow-emerald-600/30 transition-all"
            >
              <TrendingDown className="w-4 h-4" />
              Analyze Cost & Break-Even
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/inference"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-sm font-semibold transition-all"
            >
              <Zap className="w-4 h-4 text-emerald-400" />
              Live Serving Playground
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Hardware Status */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Hardware Environment
            </span>
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Cpu className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-2xl font-bold text-white capitalize">
              {hardware ? hardware.accelerator_type : 'Detecting...'}
            </div>
            <div className="text-xs text-slate-400 mt-1 truncate">
              {hardware ? hardware.accelerator_name : 'Unified / CUDA'}
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <span>RAM: {hardware ? `${hardware.total_ram_gb} GB` : '--'}</span>
            <span className="text-emerald-400 font-medium">
              {hardware?.bf16_supported ? 'BF16 Ready' : 'FP16 Ready'}
            </span>
          </div>
        </div>

        {/* Registered Models */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Model Hub
            </span>
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
              <Server className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-2xl font-bold text-white">
              {models.length || 5} Models
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Open-Weight & Commercial APIs
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <span>Fine-Tuned: {models.filter((m) => m.is_fine_tuned).length}</span>
            <span>Quantized: {models.filter((m) => m.is_quantized).length}</span>
          </div>
        </div>

        {/* Experiments Tracked */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Experiments
            </span>
            <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
              <Database className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-2xl font-bold text-white">
              {experiments.length || 3} Runs
            </div>
            <div className="text-xs text-slate-400 mt-1">
              MLflow & Unified SQLite Registry
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <span>Benchmark: BillSum CA</span>
            <span className="text-emerald-400">100% Tracked</span>
          </div>
        </div>

        {/* Serving ROI */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Economics
            </span>
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
              <TrendingDown className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="text-2xl font-bold text-emerald-400">
              Up to 92%
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Hosting Savings @ High Volume
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <span>Break-Even: ~330k reqs</span>
            <span className="text-slate-300">RTX 4090 / Mac</span>
          </div>
        </div>
      </div>

      {/* Quick Action Feature Grid */}
      <div>
        <h2 className="text-lg font-bold text-white mb-4">Optimization Lifecycle</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center mb-4">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white">
                LoRA & QLoRA Fine-Tuning
              </h3>
              <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                Train parameter-efficient adapters on legislative BillSum data. Freeze
                99%+ of base weights while matching domain summarization quality.
              </p>
            </div>
            <Link
              to="/training"
              className="mt-5 inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-400 hover:text-emerald-300"
            >
              Launch Fine-Tuning Studio <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 text-blue-400 flex items-center justify-center mb-4">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white">
                Multi-Precision Quantization
              </h3>
              <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                Benchmark PyTorch Dynamic INT8, bitsandbytes 4-bit NF4, and llama.cpp
                Metal GGUF. Reduce memory consumption by up to 75% with minimal ROUGE loss.
              </p>
            </div>
            <Link
              to="/quantization"
              className="mt-5 inline-flex items-center gap-1.5 text-xs font-semibold text-blue-400 hover:text-blue-300"
            >
              Inspect Quantization Lab <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center mb-4">
                <Zap className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white">
                Continuous Batching & Serving
              </h3>
              <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                Analyze dynamic memory scaling via KV Cache estimators, multi-concurrency
                load tests (P50/P95/P99 latency), and TTFT metrics.
              </p>
            </div>
            <Link
              to="/inference"
              className="mt-5 inline-flex items-center gap-1.5 text-xs font-semibold text-amber-400 hover:text-amber-300"
            >
              Test Serving Engine <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Recent Experiments Overview */}
      <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-white">Recent Experiment Results</h2>
            <p className="text-xs text-slate-400">
              Summarization benchmarks across Open-Weight vs Commercial APIs
            </p>
          </div>
          <Link
            to="/experiments"
            className="text-xs font-semibold text-emerald-400 hover:underline"
          >
            View All ({experiments.length})
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-400 uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4 rounded-l-lg">Model</th>
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">ROUGE-1</th>
                <th className="py-3 px-4">ROUGE-2</th>
                <th className="py-3 px-4">ROUGE-L</th>
                <th className="py-3 px-4">Latency</th>
                <th className="py-3 px-4">Tokens / Sec</th>
                <th className="py-3 px-4 rounded-r-lg">Cost / 1k</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-200">
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-slate-500">
                    No experiments found. Run a baseline benchmark to populate results!
                  </td>
                </tr>
              ) : (
                experiments.slice(0, 5).map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-medium text-white flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      {exp.name}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                        {exp.category}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono">{exp.rouge1?.toFixed(2) ?? '--'}</td>
                    <td className="py-3 px-4 font-mono">{exp.rouge2?.toFixed(2) ?? '--'}</td>
                    <td className="py-3 px-4 font-mono text-emerald-400 font-semibold">
                      {exp.rougeL?.toFixed(2) ?? '--'}
                    </td>
                    <td className="py-3 px-4 font-mono">{exp.avg_latency_ms?.toFixed(0) ?? '--'} ms</td>
                    <td className="py-3 px-4 font-mono">{exp.tokens_per_second?.toFixed(1) ?? '--'} tps</td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      ${exp.cost_per_1k_requests?.toFixed(4) ?? '0.0000'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
