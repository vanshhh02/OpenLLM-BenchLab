import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { ModelsPage } from './pages/ModelsPage';
import { FineTuningPage } from './pages/FineTuningPage';
import { QuantizationPage } from './pages/QuantizationPage';
import { InferencePage } from './pages/InferencePage';
import { CostAnalysisPage } from './pages/CostAnalysisPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { WorkspaceAuthPage } from './pages/WorkspaceAuthPage';
import { api } from './services/api';
import { HardwareProfile } from './types';

export const App: React.FC = () => {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [healthy, setHealthy] = useState<boolean>(true);

  useEffect(() => {
    async function checkHealth() {
      try {
        const [health, hw] = await Promise.all([api.getHealth(), api.getHardware()]);
        setHealthy(health.status === 'healthy');
        setHardware(hw);
      } catch (err) {
        console.warn('Backend server connecting / not reachable:', err);
        setHealthy(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Router>
      <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100 font-sans">
        <Navbar hardware={hardware} healthy={healthy} />

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/training" element={<FineTuningPage />} />
            <Route path="/quantization" element={<QuantizationPage />} />
            <Route path="/inference" element={<InferencePage />} />
            <Route path="/cost" element={<CostAnalysisPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/workspace" element={<WorkspaceAuthPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="border-t border-slate-800/80 bg-[#0f172a] py-6 text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between gap-4">
            <div>
              <span className="font-bold text-slate-400">OpenLLM BenchLab</span> — Production
              LLM Optimization & Benchmark Suite
            </div>
            <div className="flex items-center gap-4">
              <span>FastAPI Backend + MLflow</span>
              <span>•</span>
              <span>Apple MPS & CUDA</span>
              <span>•</span>
              <a
                href="https://github.com/vanshhh02/OpenLLM-BenchLab"
                target="_blank"
                rel="noreferrer"
                className="text-emerald-400 hover:underline"
              >
                GitHub Repository
              </a>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
};

export default App;

