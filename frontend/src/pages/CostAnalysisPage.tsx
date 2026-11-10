import React, { useEffect, useState } from 'react';
import {
  TrendingDown,
  DollarSign,
  Zap,
  Server,
  Calculator,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  HelpCircle,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';
import { CostCalculationResponse, SensitivityPoint } from '../types';

export const CostAnalysisPage: React.FC = () => {
  // Input parameters
  const [monthlyRequests, setMonthlyRequests] = useState<number>(500000);
  const [avgInputTokens, setAvgInputTokens] = useState<number>(1500);
  const [avgOutputTokens, setAvgOutputTokens] = useState<number>(250);

  // API presets
  const [selectedApiPreset, setSelectedApiPreset] = useState<string>('gpt-4o-mini');
  const [apiName, setApiName] = useState<string>('OpenAI GPT-4o-mini');
  const [apiInputCostPerM, setApiInputCostPerM] = useState<number>(0.150);
  const [apiOutputCostPerM, setApiOutputCostPerM] = useState<number>(0.600);

  // Dedicated Hardware presets
  const [selectedHwPreset, setSelectedHwPreset] = useState<string>('cloud_rtx4090');
  const [serverMonthlyRental, setServerMonthlyRental] = useState<number>(85.0);
  const [hardwarePurchaseCost, setHardwarePurchaseCost] = useState<number>(0.0);
  const [amortizationMonths, setAmortizationMonths] = useState<number>(36);
  const [serverPowerWatts, setServerPowerWatts] = useState<number>(300.0);
  const [electricityKwhCost, setElectricityKwhCost] = useState<number>(0.15);
  const [tokensPerSecond, setTokensPerSecond] = useState<number>(75.0);
  const [operationalOverhead, setOperationalOverhead] = useState<number>(25.0);

  // Results & Presets state
  const [result, setResult] = useState<CostCalculationResponse | null>(null);
  const [frontierPresets, setFrontierPresets] = useState<Record<string, any>>({});
  const [hardwarePresets, setHardwarePresets] = useState<Record<string, any>>({});
  const [paretoData, setParetoData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Load presets on mount
  useEffect(() => {
    async function loadPresets() {
      try {
        const [fp, hp, pareto] = await Promise.allSettled([
          api.getFrontierPresets(),
          api.getHardwarePresets(),
          api.getParetoAnalysis(),
        ]);
        if (fp.status === 'fulfilled') setFrontierPresets(fp.value);
        if (hp.status === 'fulfilled') setHardwarePresets(hp.value);
        if (pareto.status === 'fulfilled') setParetoData(pareto.value);
      } catch (e) {
        console.error('Failed to load presets:', e);
      }
    }
    loadPresets();
  }, []);

  // Update when API preset changes
  const handleApiPresetChange = (presetKey: string) => {
    setSelectedApiPreset(presetKey);
    const p = frontierPresets[presetKey];
    if (p) {
      setApiName(p.name);
      setApiInputCostPerM(p.input_cost_per_m);
      setApiOutputCostPerM(p.output_cost_per_m);
    }
  };

  // Update when Hardware preset changes
  const handleHardwarePresetChange = (presetKey: string) => {
    setSelectedHwPreset(presetKey);
    const h = hardwarePresets[presetKey];
    if (h) {
      setServerMonthlyRental(h.server_monthly_rental_usd);
      setHardwarePurchaseCost(h.hardware_purchase_cost_usd);
      setServerPowerWatts(h.power_watts);
      setTokensPerSecond(h.estimated_tps);
      setAmortizationMonths(h.amortization_months || 36);
    }
  };

  // Recalculate
  useEffect(() => {
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await api.calculateCost({
          monthly_requests: monthlyRequests,
          avg_input_tokens: avgInputTokens,
          avg_output_tokens: avgOutputTokens,
          api_name: apiName,
          api_input_cost_per_m: apiInputCostPerM,
          api_output_cost_per_m: apiOutputCostPerM,
          server_monthly_rental_usd: serverMonthlyRental,
          hardware_purchase_cost_usd: hardwarePurchaseCost,
          amortization_months: amortizationMonths,
          server_power_watts: serverPowerWatts,
          electricity_kwh_cost_usd: electricityKwhCost,
          operational_overhead_monthly_usd: operationalOverhead,
          tokens_per_second_local: tokensPerSecond,
        });
        setResult(res);
      } catch (err) {
        console.error('Cost calculation error:', err);
      } finally {
        setLoading(false);
      }
    }, 150);

    return () => clearTimeout(timer);
  }, [
    monthlyRequests,
    avgInputTokens,
    avgOutputTokens,
    apiName,
    apiInputCostPerM,
    apiOutputCostPerM,
    serverMonthlyRental,
    hardwarePurchaseCost,
    amortizationMonths,
    serverPowerWatts,
    electricityKwhCost,
    operationalOverhead,
    tokensPerSecond,
  ]);

  return (
    <div className="space-y-8">
      {/* Page Title */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider">
          <DollarSign className="w-4 h-4" />
          Phase 6: Cost Analysis & Frontier ROI Modeling
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
          TCO Calculator & Break-Even Analysis
        </h1>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Quantitatively compare total cost of ownership between frontier commercial APIs
          and self-hosted / cloud-dedicated open-weight LLMs with full amortization,
          electricity wattage, and volume sensitivity curves.
        </p>
      </div>

      {/* Main Grid: Parameters vs Comparison Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Controls */}
        <div className="lg:col-span-5 space-y-5">
          {/* Workload Volume Card */}
          <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Calculator className="w-4 h-4 text-emerald-400" />
              Inference Workload Parameters
            </h2>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Monthly Inferences:</span>
                <span className="font-mono text-emerald-400 font-bold">
                  {monthlyRequests.toLocaleString()} reqs
                </span>
              </div>
              <input
                type="range"
                min="1000"
                max="5000000"
                step="5000"
                value={monthlyRequests}
                onChange={(e) => setMonthlyRequests(Number(e.target.value))}
                className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                <span>1k</span>
                <span>500k</span>
                <span>2M</span>
                <span>5M</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Avg Input (Prompt)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={avgInputTokens}
                    onChange={(e) => setAvgInputTokens(Number(e.target.value))}
                    className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
                  />
                  <span className="absolute right-2 top-2 text-[10px] text-slate-500">
                    toks
                  </span>
                </div>
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Avg Output (Summary)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={avgOutputTokens}
                    onChange={(e) => setAvgOutputTokens(Number(e.target.value))}
                    className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
                  />
                  <span className="absolute right-2 top-2 text-[10px] text-slate-500">
                    toks
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Frontier API Configuration Card */}
          <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-blue-400" />
                Frontier API Pricing
              </h2>
              <span className="text-[10px] text-slate-400">Rate / 1M Tokens</span>
            </div>

            <div>
              <label className="block text-[11px] text-slate-400 mb-1">
                Select Model Preset
              </label>
              <select
                value={selectedApiPreset}
                onChange={(e) => handleApiPresetChange(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-200 focus:border-blue-500 focus:outline-none"
              >
                {Object.entries(frontierPresets).map(([k, p]: [string, any]) => (
                  <option key={k} value={k}>
                    {p.name} (${p.input_cost_per_m} in / ${p.output_cost_per_m} out)
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Input Rate ($/1M)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={apiInputCostPerM}
                  onChange={(e) => setApiInputCostPerM(Number(e.target.value))}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Output Rate ($/1M)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={apiOutputCostPerM}
                  onChange={(e) => setApiOutputCostPerM(Number(e.target.value))}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Self-Hosted Infrastructure Card */}
          <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Server className="w-4 h-4 text-emerald-400" />
              Self-Hosted / Cloud Hardware
            </h2>

            <div>
              <label className="block text-[11px] text-slate-400 mb-1">
                Hardware Preset
              </label>
              <select
                value={selectedHwPreset}
                onChange={(e) => handleHardwarePresetChange(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-200 focus:border-emerald-500 focus:outline-none"
              >
                {Object.entries(hardwarePresets).map(([k, h]: [string, any]) => (
                  <option key={k} value={k}>
                    {h.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Monthly Cloud Rental ($)
                </label>
                <input
                  type="number"
                  value={serverMonthlyRental}
                  onChange={(e) => setServerMonthlyRental(Number(e.target.value))}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Purchase Cost ($)
                </label>
                <input
                  type="number"
                  value={hardwarePurchaseCost}
                  onChange={(e) => setHardwarePurchaseCost(Number(e.target.value))}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Power Draw (Watts)
                </label>
                <input
                  type="number"
                  value={serverPowerWatts}
                  onChange={(e) => setServerPowerWatts(Number(e.target.value))}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Serving Speed (TPS)
                </label>
                <input
                  type="number"
                  value={tokensPerSecond}
                  onChange={(e) => setTokensPerSecond(Number(e.target.value))}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-white font-mono focus:border-emerald-500 focus:outline-none"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Comparative Results & Economics */}
        <div className="lg:col-span-7 space-y-5">
          {/* Primary ROI Summary Card */}
          <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 border border-slate-700/80 shadow-xl space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700/80 pb-4">
              <div>
                <span className="text-xs text-slate-400 font-medium">Monthly Comparison</span>
                <div className="text-lg font-bold text-white">
                  {monthlyRequests.toLocaleString()} Requests / Month
                </div>
              </div>
              <div
                className={`px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 ${
                  result && result.monthly_savings_usd > 0
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                }`}
              >
                {result && result.monthly_savings_usd > 0 ? (
                  <>
                    <CheckCircle className="w-3.5 h-3.5" /> Self-Hosted Wins
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" /> Frontier API Wins
                  </>
                )}
              </div>
            </div>

            {/* Side-by-side cost boxes */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* API Cost Box */}
              <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <span className="text-xs text-slate-400 block">{apiName}</span>
                <div className="text-2xl font-extrabold text-white mt-1">
                  ${result ? result.monthly_api_cost_usd.toLocaleString(undefined, { minimumFractionDigits: 2 }) : '--'}
                  <span className="text-xs font-normal text-slate-400"> /mo</span>
                </div>
                <div className="mt-3 text-xs text-slate-300 space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Per Request:</span>
                    <span className="font-mono">${result?.api_cost_per_request_usd.toFixed(6)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Per 1,000 Reqs:</span>
                    <span className="font-mono font-medium text-slate-200">
                      ${result?.api_cost_per_1k_usd.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Annual Projected:</span>
                    <span className="font-mono">${result ? (result.monthly_api_cost_usd * 12).toLocaleString() : '--'}</span>
                  </div>
                </div>
              </div>

              {/* Local Dedicated Box */}
              <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <span className="text-xs text-slate-400 block">Self-Hosted Dedicated</span>
                <div className="text-2xl font-extrabold text-emerald-400 mt-1">
                  ${result ? result.monthly_local_cost_usd.toLocaleString(undefined, { minimumFractionDigits: 2 }) : '--'}
                  <span className="text-xs font-normal text-slate-400"> /mo</span>
                </div>
                <div className="mt-3 text-xs text-slate-300 space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Per Request:</span>
                    <span className="font-mono">${result?.local_cost_per_request_usd.toFixed(6)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Per 1,000 Reqs:</span>
                    <span className="font-mono font-medium text-emerald-400">
                      ${result?.local_cost_per_1k_usd.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Power Consumed:</span>
                    <span className="font-mono">${result?.local_electricity_cost_monthly_usd.toFixed(2)}/mo</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Savings & Break-Even Metric Callout */}
            <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/40 flex flex-wrap items-center justify-between gap-4">
              <div>
                <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
                  Net Monthly Difference
                </span>
                <div
                  className={`text-xl font-bold ${
                    result && result.monthly_savings_usd > 0 ? 'text-emerald-400' : 'text-blue-400'
                  }`}
                >
                  {result && result.monthly_savings_usd > 0
                    ? `+$${result.monthly_savings_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })} (${result.savings_percentage}% saved)`
                    : `API is $${Math.abs(result?.monthly_savings_usd ?? 0).toFixed(2)} cheaper`}
                </div>
              </div>

              <div className="text-right">
                <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
                  Break-Even Threshold
                </span>
                <div className="text-base font-mono font-bold text-white">
                  {result?.break_even_requests
                    ? `${Math.round(result.break_even_requests).toLocaleString()} reqs/mo`
                    : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Volume Sensitivity Table */}
          <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
            <h2 className="text-sm font-bold text-white mb-2 flex items-center justify-between">
              <span>Volume Sensitivity Curve</span>
              <span className="text-[11px] text-slate-400 font-normal">
                Monthly TCO Scaling
              </span>
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-800/80 text-slate-400">
                  <tr>
                    <th className="py-2.5 px-3 rounded-l-lg">Monthly Volume</th>
                    <th className="py-2.5 px-3">API Bill</th>
                    <th className="py-2.5 px-3">Self-Hosted</th>
                    <th className="py-2.5 px-3">Savings</th>
                    <th className="py-2.5 px-3 rounded-r-lg">Recommended</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-300">
                  {result?.sensitivity_curve.map((pt: SensitivityPoint) => {
                    const isSelfHosted = pt.recommended_architecture === 'self_hosted';
                    return (
                      <tr
                        key={pt.monthly_requests}
                        className={`hover:bg-slate-800/40 transition-colors ${
                          pt.monthly_requests === monthlyRequests
                            ? 'bg-emerald-500/10 font-semibold'
                            : ''
                        }`}
                      >
                        <td className="py-2 px-3 font-mono">
                          {pt.monthly_requests.toLocaleString()}
                        </td>
                        <td className="py-2 px-3 font-mono">
                          ${pt.api_monthly_cost_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </td>
                        <td className="py-2 px-3 font-mono">
                          ${pt.local_monthly_cost_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </td>
                        <td
                          className={`py-2 px-3 font-mono ${
                            pt.savings_usd > 0 ? 'text-emerald-400' : 'text-slate-400'
                          }`}
                        >
                          {pt.savings_usd > 0
                            ? `+$${pt.savings_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                            : `-$${Math.abs(pt.savings_usd).toFixed(2)}`}
                        </td>
                        <td className="py-2 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                              isSelfHosted
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                            }`}
                          >
                            {isSelfHosted ? 'Self-Hosted' : 'Frontier API'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pareto Frontier Insights Card */}
          {paretoData && (
            <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
              <h2 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-emerald-400" />
                Pareto-Optimal Models (Quality vs. Cost vs. Latency)
              </h2>
              <p className="text-xs text-slate-400 mb-3">
                Models that are non-dominated across ROUGE-L, serving cost, and latency:
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {paretoData.pareto_frontier.map((p: any) => (
                  <div
                    key={p.id}
                    className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60 text-xs"
                  >
                    <div className="font-bold text-white truncate">{p.name}</div>
                    <div className="text-[11px] text-slate-400 capitalize">{p.category}</div>
                    <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px]">
                      <div>
                        <span className="text-slate-500 block text-[9px]">ROUGE-L</span>
                        <span className="text-emerald-400 font-bold">{p.rougeL?.toFixed(1)}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">LATENCY</span>
                        <span>{p.avg_latency_ms?.toFixed(0)} ms</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">COST/1K</span>
                        <span>${p.cost_per_1k_requests?.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
