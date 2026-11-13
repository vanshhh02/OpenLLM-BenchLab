import React, { useCallback, useEffect, useState } from 'react';
import {
  Shield,
  Users,
  Key,
  Layers,
  LogIn,
  LogOut,
  UserPlus,
  Plus,
  Trash2,
  Copy,
  Check,
  RefreshCw,
  XCircle,
  Play,
  Server,
} from 'lucide-react';
import { api, authStorage } from '../services/api';
import {
  ApiKeyItem,
  AsyncJobItem,
  AuthMeResponse,
  WorkerHealthResponse,
  WorkspaceMemberItem,
  WorkspaceSummary,
} from '../types';

export const WorkspaceAuthPage: React.FC = () => {
  const [me, setMe] = useState<AuthMeResponse | null>(null);
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [members, setMembers] = useState<WorkspaceMemberItem[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKeyItem[]>([]);
  const [jobs, setJobs] = useState<AsyncJobItem[]>([]);
  const [workerHealth, setWorkerHealth] = useState<WorkerHealthResponse | null>(null);

  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [workspaceName, setWorkspaceName] = useState('');

  const [newWsName, setNewWsName] = useState('');
  const [newWsDesc, setNewWsDesc] = useState('');
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<'admin' | 'member' | 'viewer'>('member');

  const [newKeyName, setNewKeyName] = useState('');
  const [createdRawKey, setCreatedRawKey] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState(false);

  const [jobType, setJobType] = useState<'evaluation' | 'quantization' | 'load_test' | 'training'>('evaluation');
  const [jobModelId, setJobModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const activeWorkspaceId = authStorage.getWorkspaceId();

  const refreshAll = useCallback(async () => {
    try {
      setErrorMsg(null);
      const [meRes, wsList, healthRes, jobList] = await Promise.all([
        api.getMe(),
        api.getWorkspaces(),
        api.getWorkerHealth(),
        api.getJobs(),
      ]);
      setMe(meRes);
      setWorkspaces(wsList);
      setWorkerHealth(healthRes);
      setJobs(jobList);

      if (meRes.authenticated) {
        const [memberRes, keyRes] = await Promise.all([
          api.getWorkspaceMembers(meRes.workspace_id).catch(() => []),
          api.getApiKeys().catch(() => []),
        ]);
        setMembers(memberRes);
        setApiKeys(keyRes);
      } else {
        setMembers([]);
        setApiKeys([]);
      }
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to load workspace & auth telemetry.');
    }
  }, []);

  useEffect(() => {
    refreshAll();
    const timer = setInterval(() => {
      api.getJobs().then(setJobs).catch(() => {});
      api.getWorkerHealth().then(setWorkerHealth).catch(() => {});
    }, 5000);
    return () => clearInterval(timer);
  }, [refreshAll]);

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    setStatusMsg(null);
    try {
      if (authMode === 'register') {
        await api.register({
          email,
          password,
          full_name: fullName,
          workspace_name: workspaceName || undefined,
        });
        setStatusMsg('Account & personal workspace created.');
      } else {
        await api.login({ email, password });
        setStatusMsg('Signed in successfully.');
      }
      setPassword('');
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Authentication failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    authStorage.clear();
    setCreatedRawKey(null);
    setStatusMsg('Signed out — reverted to local default workspace.');
    await refreshAll();
  };

  const handleSwitchWorkspace = async (wsId: string) => {
    authStorage.setWorkspaceId(wsId);
    setStatusMsg(`Switched active workspace to ${wsId}.`);
    await refreshAll();
  };

  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWsName.trim()) return;
    try {
      setErrorMsg(null);
      const ws = await api.createWorkspace({ name: newWsName, description: newWsDesc || undefined });
      authStorage.setWorkspaceId(ws.id);
      setNewWsName('');
      setNewWsDesc('');
      setStatusMsg(`Created workspace "${ws.name}".`);
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to create workspace.');
    }
  };

  const handleAddMember = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) return;
    try {
      setErrorMsg(null);
      await api.addWorkspaceMember(activeWorkspaceId, { email: inviteEmail, role: inviteRole });
      setInviteEmail('');
      setStatusMsg(`Added ${inviteEmail} as ${inviteRole}.`);
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to add workspace member.');
    }
  };

  const handleCreateApiKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;
    try {
      setErrorMsg(null);
      const created = await api.createApiKey({ name: newKeyName, scopes: ['read', 'write'] });
      setCreatedRawKey(created.raw_key || null);
      setCopiedKey(false);
      setNewKeyName('');
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to create API key.');
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    try {
      await api.revokeApiKey(keyId);
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to revoke API key.');
    }
  };

  const handleSubmitJob = async () => {
    try {
      setErrorMsg(null);
      const payload =
        jobType === 'evaluation'
          ? { model_id: jobModelId, dataset_name: 'billsum', split: 'ca_test', num_samples: 3 }
          : jobType === 'quantization'
          ? { base_model_id: jobModelId, quantization_method: 'dynamic_int8', target_precision: 'int8', eval_samples: 2 }
          : jobType === 'load_test'
          ? { model_id: jobModelId, concurrency_levels: [1, 2], requests_per_level: 2 }
          : { base_model_id: jobModelId, method: 'lora', max_steps: 5 };

      const job = await api.submitJob({ job_type: jobType, payload });
      setStatusMsg(`Queued async job ${job.id} (${job.backend}).`);
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to enqueue job.');
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await api.cancelJob(jobId);
      await refreshAll();
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to cancel job.');
    }
  };

  const copyRawKey = () => {
    if (!createdRawKey) return;
    navigator.clipboard.writeText(createdRawKey);
    setCopiedKey(true);
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
            <Shield className="w-6 h-6 text-emerald-400" />
            Auth, Multi-User Workspaces & Async Job Queue
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Manage JWT authentication, scoped API keys, workspace RBAC isolation, and background workers.
          </p>
        </div>

        <button
          onClick={refreshAll}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh State
        </button>
      </div>

      {errorMsg && (
        <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm">
          {errorMsg}
        </div>
      )}
      {statusMsg && (
        <div className="p-3.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-sm">
          {statusMsg}
        </div>
      )}

      {/* Identity & Workspace Switcher Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Authentication Card */}
        <div className="rounded-xl bg-[#0f172a] border border-slate-800 p-6 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              {me?.authenticated ? <LogIn className="w-5 h-5 text-emerald-400" /> : <UserPlus className="w-5 h-5 text-emerald-400" />}
              Identity & Session
            </h2>
            <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700">
              {me ? `${me.auth_method.toUpperCase()} • ${me.workspace_role.toUpperCase()}` : 'LOADING'}
            </span>
          </div>

          {me?.authenticated && me.user ? (
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-slate-900/70 border border-slate-800 space-y-1">
                <div className="text-sm font-medium text-white">{me.user.full_name || me.user.email}</div>
                <div className="text-xs text-slate-400">{me.user.email}</div>
                <div className="text-xs text-emerald-400 font-mono pt-1">
                  Active Workspace: {me.workspace_id}
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 border border-rose-500/40 text-rose-300 text-sm font-medium"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </div>
          ) : (
            <form onSubmit={handleAuthSubmit} className="space-y-3">
              <div className="flex gap-2 p-1 rounded-lg bg-slate-900 border border-slate-800">
                <button
                  type="button"
                  onClick={() => setAuthMode('login')}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    authMode === 'login' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => setAuthMode('register')}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    authMode === 'register' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Create Account
                </button>
              </div>

              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email address"
                className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-white focus:border-emerald-500 focus:outline-none"
              />
              <input
                type="password"
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password (min 6 chars)"
                className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-white focus:border-emerald-500 focus:outline-none"
              />
              {authMode === 'register' && (
                <>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Full name"
                    className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  />
                  <input
                    type="text"
                    value={workspaceName}
                    onChange={(e) => setWorkspaceName(e.target.value)}
                    placeholder="Initial workspace name (optional)"
                    className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  />
                </>
              )}
              <button
                type="submit"
                disabled={loading}
                className="w-full py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium transition-colors"
              >
                {loading ? 'Processing...' : authMode === 'register' ? 'Register & Provision Workspace' : 'Sign In with JWT'}
              </button>
            </form>
          )}
        </div>

        {/* Workspaces & Members Card */}
        <div className="rounded-xl bg-[#0f172a] border border-slate-800 p-6 space-y-5">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Users className="w-5 h-5 text-emerald-400" />
            Workspaces & RBAC
          </h2>

          <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
            {workspaces.map((ws) => {
              const isActive = ws.id === (me?.workspace_id || activeWorkspaceId);
              return (
                <div
                  key={ws.id}
                  onClick={() => handleSwitchWorkspace(ws.id)}
                  className={`cursor-pointer p-3 rounded-lg border flex items-center justify-between transition-colors ${
                    isActive
                      ? 'bg-emerald-500/10 border-emerald-500/40'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div>
                    <div className="text-sm font-medium text-white">{ws.name}</div>
                    <div className="text-xs text-slate-400 font-mono">
                      {ws.id} • {ws.member_count} member(s)
                    </div>
                  </div>
                  <span className="px-2 py-0.5 rounded text-xs font-mono bg-slate-800 text-emerald-400">
                    {ws.role}
                  </span>
                </div>
              );
            })}
          </div>

          {me?.authenticated && (
            <form onSubmit={handleCreateWorkspace} className="flex gap-2 pt-2 border-t border-slate-800">
              <input
                type="text"
                value={newWsName}
                onChange={(e) => setNewWsName(e.target.value)}
                placeholder="New workspace name"
                className="flex-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white focus:border-emerald-500 focus:outline-none"
              />
              <button
                type="submit"
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-medium text-white"
              >
                <Plus className="w-3.5 h-3.5" />
                Create
              </button>
            </form>
          )}

          {me?.authenticated && (
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Workspace Members ({members.length})
              </div>
              <div className="space-y-1.5 max-h-32 overflow-y-auto">
                {members.map((m) => (
                  <div key={m.id} className="flex items-center justify-between text-xs bg-slate-900/50 px-3 py-2 rounded-lg">
                    <div>
                      <span className="text-white font-medium">{m.email}</span>
                      <span className="text-slate-500 ml-2">({m.role})</span>
                    </div>
                    {m.role !== 'owner' && (
                      <button
                        onClick={() => api.removeWorkspaceMember(activeWorkspaceId, m.user_id).then(refreshAll)}
                        className="text-rose-400 hover:text-rose-300"
                        title="Remove member"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <form onSubmit={handleAddMember} className="flex gap-2">
                <input
                  type="email"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  placeholder="Add registered user by email"
                  className="flex-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white focus:border-emerald-500 focus:outline-none"
                />
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value as any)}
                  className="px-2 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
                >
                  <option value="viewer">viewer</option>
                  <option value="member">member</option>
                  <option value="admin">admin</option>
                </select>
                <button
                  type="submit"
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-white"
                >
                  Add
                </button>
              </form>
            </div>
          )}
        </div>
      </div>

      {/* API Keys Card */}
      <div className="rounded-xl bg-[#0f172a] border border-slate-800 p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Key className="w-5 h-5 text-emerald-400" />
            Scoped API Keys
          </h2>
          {me?.authenticated && (
            <form onSubmit={handleCreateApiKey} className="flex gap-2">
              <input
                type="text"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                placeholder="Key label (e.g. CI Runner)"
                className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white focus:border-emerald-500 focus:outline-none"
              />
              <button
                type="submit"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-medium text-white"
              >
                <Plus className="w-3.5 h-3.5" />
                Generate Key
              </button>
            </form>
          )}
        </div>

        {createdRawKey && (
          <div className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between gap-4">
            <div className="text-xs text-amber-200">
              <span className="font-semibold">Copy your new API key now</span> (shown once):{' '}
              <code className="font-mono bg-slate-950 px-2 py-0.5 rounded text-emerald-300">
                {createdRawKey}
              </code>
            </div>
            <button
              onClick={copyRawKey}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-white"
            >
              {copiedKey ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copiedKey ? 'Copied' : 'Copy'}
            </button>
          </div>
        )}

        {!me?.authenticated ? (
          <p className="text-xs text-slate-400">Sign in to create and manage scoped API keys for CLI/CI pipelines.</p>
        ) : apiKeys.length === 0 ? (
          <p className="text-xs text-slate-400">No API keys created in this workspace yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2 pr-4">Name</th>
                  <th className="py-2 pr-4">Prefix</th>
                  <th className="py-2 pr-4">Scopes</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {apiKeys.map((k) => (
                  <tr key={k.id}>
                    <td className="py-2.5 pr-4 text-white font-medium">{k.name}</td>
                    <td className="py-2.5 pr-4 font-mono text-slate-300">{k.key_prefix}...</td>
                    <td className="py-2.5 pr-4 text-slate-400">{k.scopes.join(', ')}</td>
                    <td className="py-2.5 pr-4">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                          k.is_active ? 'bg-emerald-500/15 text-emerald-400' : 'bg-slate-800 text-slate-500'
                        }`}
                      >
                        {k.is_active ? 'active' : 'revoked'}
                      </span>
                    </td>
                    <td className="py-2.5 text-right">
                      {k.is_active && (
                        <button
                          onClick={() => handleRevokeKey(k.id)}
                          className="text-rose-400 hover:text-rose-300 font-medium"
                        >
                          Revoke
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Async Job Queue & Worker Monitor */}
      <div className="rounded-xl bg-[#0f172a] border border-slate-800 p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-emerald-400" />
              Async Job Queue & Background Workers
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 flex items-center gap-2">
              <Server className="w-3.5 h-3.5 text-slate-500" />
              Backend: <code className="text-emerald-400">{workerHealth?.backend || 'local_threadpool'}</code> •
              Total Jobs: {workerHealth?.total_jobs ?? jobs.length}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={jobType}
              onChange={(e) => setJobType(e.target.value as any)}
              className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
            >
              <option value="evaluation">Evaluation</option>
              <option value="quantization">Quantization</option>
              <option value="load_test">Load Test</option>
              <option value="training">LoRA Training</option>
            </select>
            <input
              type="text"
              value={jobModelId}
              onChange={(e) => setJobModelId(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white w-56"
            />
            <button
              onClick={handleSubmitJob}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-medium text-white"
            >
              <Play className="w-3.5 h-3.5" />
              Enqueue Job
            </button>
          </div>
        </div>

        {jobs.length === 0 ? (
          <p className="text-xs text-slate-400">No async jobs queued in this workspace yet.</p>
        ) : (
          <div className="space-y-2.5">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="p-3.5 rounded-lg bg-slate-900/70 border border-slate-800 flex flex-wrap items-center justify-between gap-4"
              >
                <div className="space-y-1 min-w-[220px]">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-white">{job.id}</span>
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-[11px] text-slate-300">
                      {job.job_type}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                        job.status === 'completed'
                          ? 'bg-emerald-500/15 text-emerald-400'
                          : job.status === 'failed'
                          ? 'bg-rose-500/15 text-rose-400'
                          : job.status === 'cancelled'
                          ? 'bg-slate-700 text-slate-400'
                          : 'bg-amber-500/15 text-amber-300'
                      }`}
                    >
                      {job.status}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400">
                    {job.error_message ? `Error: ${job.error_message}` : job.current_step || 'Pending'}
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="w-36">
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 transition-all"
                        style={{ width: `${Math.min(100, Math.max(0, job.progress_pct))}%` }}
                      />
                    </div>
                    <div className="text-[11px] text-slate-400 text-right mt-1">{job.progress_pct}%</div>
                  </div>
                  {(job.status === 'queued' || job.status === 'running') && (
                    <button
                      onClick={() => handleCancelJob(job.id)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 text-xs"
                    >
                      <XCircle className="w-3.5 h-3.5" />
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

