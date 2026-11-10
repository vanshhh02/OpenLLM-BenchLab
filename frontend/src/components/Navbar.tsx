import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Activity,
  Cpu,
  Layers,
  Zap,
  TrendingDown,
  Database,
  BarChart3,
  Server,
  DollarSign,
} from 'lucide-react';
import { HardwareProfile } from '../types';

interface NavbarProps {
  hardware: HardwareProfile | null;
  healthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ hardware, healthy }) => {
  const navItems = [
    { to: '/', label: 'Overview', icon: BarChart3 },
    { to: '/models', label: 'Models & Baseline', icon: Server },
    { to: '/training', label: 'Fine-Tuning (LoRA)', icon: Layers },
    { to: '/quantization', label: 'Quantization', icon: Cpu },
    { to: '/inference', label: 'Inference & Serving', icon: Zap },
    { to: '/cost', label: 'Cost & Frontier ROI', icon: DollarSign },
    { to: '/experiments', label: 'Experiments', icon: Database },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#0f172a]/95 backdrop-blur border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Branding */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
                OpenLLM <span className="text-emerald-400">BenchLab</span>
              </span>
              <span className="text-xs text-slate-400 block -mt-1">
                Optimization & Benchmark Suite
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </NavLink>
              );
            })}
          </nav>

          {/* Status & Accelerator Badge */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  healthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300 font-mono">
                {hardware
                  ? `${hardware.accelerator_type.toUpperCase()}: ${hardware.cpu_brand.slice(0, 18)}`
                  : 'Backend Connecting...'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
