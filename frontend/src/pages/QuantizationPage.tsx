import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Play,
  TrendingDown,
  CheckCircle2,
  HardDrive,
  Zap,
  Layers,
  ArrowRight,
} from 'lucide-react';
import { api } from '../services/api';
import { QuantizationRecord } from '../types';

export const QuantizationPage: React.FC = () => {
  const [quantizations, setQuantizations] = useState<QuantizationRecord[]>([]);
  const [precisionMatrix, setPrecisionMatrix] = useState<any[]>([]);
  const [baseModelId, setBaseModelId] = useState<string>('Qwen/Qwen2.5-0.5B-Instruct');
  const [selectedMethod, setSelectedMethod] = useState<string>('dynamic_int8');
  const [selectedPrecision, setSelectedPrecision] = useState<string>('int8');
  const [running, setRunning] = useState<boolean>(false);
  const [lastResult, setLastResult] = useState<any>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [qRes, matrixRes] = await Promise.allSettled([
          api.getQuantizations(),
          api.getPrecisionMatrix(baseModelId),
        ]);
        if (qRes.status === 'fulfilled') setQuantizations(qRes.value);
        if (matrixRes.status === 'fulfilled') setPrecisionMatrix(matrixRes.value);
      } catch (e) {
        console.error('Failed to load quantization data:', e);
      }
    }
    loadData();
  }, [baseModelId]);

  const handleRunQuantization = async () => {
    setRunning(true);
    setLastResult(null);
    try {
      const res = await api.runQuantization({
        base_model_id: baseModelId,
        method: selectedMethod,
        target_precision: selectedPrecision,
      });
      setLastResult(res);
      // Reload
      const updated = await api.getQuantizations();
      setQuantizations(updated);
    } catch (e) {
      console.error('Quantization failed:', e);
      alert('Quantization run failed. Check server logs.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-blue-400 uppercase tracking-wider">
          <Cpu className="w-4 h-4" />
          Phase 4: Multi-Precision Quantization
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
          Model Compression & Quantization Lab
        </h1>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-2xl">
          Evaluate weight and activation quantization across PyTorch Dynamic INT8,
          bitsandbytes 4-bit NF4, and llama.cpp Metal GGUF. Measure memory reduction,
          latency acceleration, and quality retention.
        </p>
      </div>

      {/* Precision Comparison Matrix Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 border border-slate-700/80 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700/80 pb-3">
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-emerald-400" />
              Precision Scaling Matrix: {baseModelId}
            </h2>
            <p className="text-xs text-slate-400">
              Benchmark comparison across weight precisions from 16-bit to 4-bit
            </p>
          </div>
          <span className="text-xs text-emerald-400 font-mono font-medium">
            Up to 75% VRAM Reduction
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/80 text-slate-400 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-2.5 px-3 rounded-l-lg">Precision Format</th>
                <th className="py-2.5 px-3">Method</th>
                <th className="py-2.5 px-3">VRAM (MB)</th>
                <th className="py-2.5 px-3">Reduction %</th>
                <th className="py-2.5 px-3">Latency (ms)</th>
                <th className="py-2.5 px-3">Speedup</th>
                <th className="py-2.5 px-3">ROUGE-L</th>
                <th className="py-2.5 px-3 rounded-r-lg">Retention %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-200">
              {precisionMatrix.map((row: any) => (
                <tr key={row.target_precision} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2.5 px-3 font-mono font-bold text-white uppercase">
                    {row.target_precision}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300 capitalize">
                    {row.method.replace('_', ' ')}
                  </td>
                  <td className="py-2.5 px-3 font-mono">{row.quantized_memory_mb.toFixed(0)} MB</td>
                  <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">
                    {row.memory_reduction_pct.toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 font-mono">{row.quantized_latency_ms.toFixed(0)} ms</td>
                  <td className="py-2.5 px-3 font-mono text-blue-400 font-semibold">
                    {row.speedup_ratio.toFixed(2)}x
                  </td>
                  <td className="py-2.5 px-3 font-mono">{row.quantized_rougel.toFixed(2)}</td>
                  <td className="py-2.5 px-3 font-mono text-emerald-400">
                    {((row.quantized_rougel / row.baseline_rougel) * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quantization Runner & Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Runner Form */}
        <div className="lg:col-span-5 p-5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <Play className="w-4 h-4 text-emerald-400" />
            Quantize Model Checkpoint
          </h2>

          <div>
            <label className="block text-[11px] text-slate-400 mb-1">Target Model</label>
            <select
              value={baseModelId}
              onChange={(e) => setBaseModelId(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
            >
              <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen/Qwen2.5-0.5B-Instruct</option>
              <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen/Qwen2.5-1.5B-Instruct</option>
              <option value="google/gemma-2-2b-it">google/gemma-2-2b-it</option>
            </select>
          </div>

          <div>
            <label className="block text-[11px] text-slate-400 mb-1">Quantization Format</label>
            <select
              value={selectedMethod}
              onChange={(e) => {
                const m = e.target.value;
                setSelectedMethod(m);
                if (m === 'dynamic_int8') setSelectedPrecision('int8');
                else if (m === 'bitsandbytes_nf4') setSelectedPrecision('nf4_4bit');
                else if (m === 'gguf_q4_k_m') setSelectedPrecision('q4_k_m');
              }}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white focus:border-emerald-500 focus:outline-none"
            >
              <option value="dynamic_int8">PyTorch Dynamic Quantization (INT8 Linear)</option>
              <option value="bitsandbytes_nf4">bitsandbytes NormalFloat4 (NF4 4-bit)</option>
              <option value="gguf_q4_k_m">llama.cpp Metal GGUF (Q4_K_M)</option>
            </select>
          </div>

          <button
            disabled={running}
            onClick={handleRunQuantization}
            className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-bold flex items-center justify-center gap-2 shadow-lg shadow-blue-600/20 transition-all"
          >
            {running ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Quantizing & Benchmarking...
              </>
            ) : (
              <>
                <Cpu className="w-4 h-4" />
                Execute Quantization Pipeline
              </>
            )}
          </button>

          {lastResult && (
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs space-y-1">
              <div className="font-bold text-emerald-400 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" /> Quantization Succeeded
              </div>
              <div className="text-slate-300 font-mono text-[11px]">
                Memory: {lastResult.original_memory_mb} MB → {lastResult.quantized_memory_mb} MB (
                {lastResult.memory_reduction_pct}% reduction)
              </div>
            </div>
          )}
        </div>

        {/* Quantization Artifact Records */}
        <div className="lg:col-span-7 p-5 rounded-xl bg-slate-900/80 border border-slate-800">
          <h2 className="text-sm font-bold text-white mb-3">
            Registered Quantization Artifacts ({quantizations.length})
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-800/80 text-slate-400">
                <tr>
                  <th className="py-2.5 px-3 rounded-l-lg">Model ID</th>
                  <th className="py-2.5 px-3">Method</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">VRAM Reduction</th>
                  <th className="py-2.5 px-3 rounded-r-lg">Speedup</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {quantizations.map((q) => (
                  <tr key={q.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2 px-3 font-mono font-medium text-white truncate max-w-[140px]">
                      {q.quantized_model_id}
                    </td>
                    <td className="py-2 px-3 capitalize">{q.method.replace('_', ' ')}</td>
                    <td className="py-2 px-3 font-mono uppercase">{q.target_precision}</td>
                    <td className="py-2 px-3 font-mono text-emerald-400 font-semibold">
                      {q.memory_reduction_pct.toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 font-mono text-blue-400">
                      {q.speedup_ratio ? `${q.speedup_ratio.toFixed(2)}x` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
