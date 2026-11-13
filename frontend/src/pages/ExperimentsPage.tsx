import React, { useEffect, useState } from 'react';
import {
  Database,
  Search,
  Filter,
  TrendingDown,
  ArrowUpDown,
  Download,
  Sparkles,
  ExternalLink,
} from 'lucide-react';
import { api } from '../services/api';
import { ExperimentRecord } from '../types';

export const ExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<keyof ExperimentRecord>('rougeL');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadExp() {
      try {
        const res = await api.getExperiments(
          categoryFilter === 'all' ? undefined : categoryFilter
        );
        setExperiments(res);
      } catch (e) {
        console.error('Failed to load experiments:', e);
      } finally {
        setLoading(false);
      }
    }
    loadExp();
  }, [categoryFilter]);

  const handleSort = (field: keyof ExperimentRecord) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const filteredExperiments = experiments
    .filter(
      (e) =>
        e.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        e.model_id.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      const valA = (a[sortField] as number) ?? 0;
      const valB = (b[sortField] as number) ?? 0;
      return sortAsc ? (valA > valB ? 1 : -1) : valA < valB ? 1 : -1;
    });

  const exportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(experiments, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `openllm_benchlab_experiments_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider">
            <Database className="w-4 h-4" />
            Unified Experiment Registry & MLflow Tracking
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Experiment Comparison & Pareto Analysis
          </h1>
          <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-2xl">
            Cross-evaluate baseline open-weight models, parameter-efficient LoRA adapters,
            quantized checkpoints, and commercial frontier APIs across standardized quality,
            throughput, and operational costs.
          </p>
        </div>

        <button
          onClick={exportJSON}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-medium transition-all"
        >
          <Download className="w-4 h-4" />
          Export JSON
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <div className="relative w-full sm:w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <input
              type="text"
              placeholder="Search model or experiment..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto w-full sm:w-auto">
          {['all', 'baseline', 'lora', 'quantized', 'commercial_api'].map((cat) => (
            <button
              key={cat}
              onClick={() => setCategoryFilter(cat)}
              className={`px-3 py-1 rounded-lg text-xs font-medium capitalize whitespace-nowrap transition-colors ${
                categoryFilter === cat
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {cat.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Experiments Table */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800 text-slate-400 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3 px-4">Experiment Name</th>
                <th className="py-3 px-4">Category</th>
                <th
                  onClick={() => handleSort('rouge1')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    ROUGE-1 <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('rouge2')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    ROUGE-2 <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('rougeL')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    ROUGE-L <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('avg_latency_ms')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    Latency <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('tokens_per_second')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    TPS <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('cost_per_1k_requests')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    Cost / 1k <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-200">
              {filteredExperiments.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-slate-500">
                    No experiments matching criteria. Run a benchmark from the Models page!
                  </td>
                </tr>
              ) : (
                filteredExperiments.map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-medium text-white">
                      <div className="font-bold">{exp.name}</div>
                      <div className="text-[10px] text-slate-500 font-mono">{exp.model_id}</div>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                          exp.category === 'commercial_api'
                            ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                            : exp.category === 'lora'
                            ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                            : exp.category === 'quantized'
                            ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        }`}
                      >
                        {exp.category}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono">{exp.rouge1?.toFixed(2) ?? '--'}</td>
                    <td className="py-3 px-4 font-mono">{exp.rouge2?.toFixed(2) ?? '--'}</td>
                    <td className="py-3 px-4 font-mono font-bold text-emerald-400">
                      {exp.rougeL?.toFixed(2) ?? '--'}
                    </td>
                    <td className="py-3 px-4 font-mono">{exp.avg_latency_ms?.toFixed(0) ?? '--'} ms</td>
                    <td className="py-3 px-4 font-mono font-semibold text-white">
                      {exp.tokens_per_second?.toFixed(1) ?? '--'}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      ${exp.cost_per_1k_requests?.toFixed(4) ?? '0.0000'}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300">
                        {exp.status}
                      </span>
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

