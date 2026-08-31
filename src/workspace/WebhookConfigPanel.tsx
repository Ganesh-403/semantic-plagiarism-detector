import React, { useState } from 'react';
import {
  Webhook,
  Plus,
  Trash2,
  ToggleLeft,
  ToggleRight,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  Clock,
  Activity,
  Zap,
  Settings,
  RefreshCw,
  AlertTriangle,
  Edit3,
  Save,
  X,
} from 'lucide-react';
import { WebhookConfig, AlertRule, WebhookDeliveryLog } from './notificationTypes';

interface WebhookConfigPanelProps {
  webhooks: WebhookConfig[];
  alertRules: AlertRule[];
  deliveryLogs: WebhookDeliveryLog[];
}

const EVENT_LABELS: Record<string, string> = {
  'scan.completed': 'Scan Completed',
  'scan.failed': 'Scan Failed',
  'plagiarism.detected': 'Plagiarism Detected',
  'document.uploaded': 'Document Uploaded',
  'document.deleted': 'Document Deleted',
  'compliance.breach': 'Compliance Breach',
  'threshold.changed': 'Threshold Changed',
  'user.role_changed': 'Role Changed',
};

const EVENT_COLORS: Record<string, string> = {
  'scan.completed': 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300',
  'scan.failed': 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  'plagiarism.detected': 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  'document.uploaded': 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300',
  'document.deleted': 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300',
  'compliance.breach': 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300',
  'threshold.changed': 'bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300',
  'user.role_changed': 'bg-pink-100 dark:bg-pink-900/40 text-pink-700 dark:text-pink-300',
};

function WebhookCard({ webhook }: { webhook: WebhookConfig }) {
  const [expanded, setExpanded] = useState(false);
  const successRate = webhook.success_count + webhook.failure_count > 0
    ? ((webhook.success_count / (webhook.success_count + webhook.failure_count)) * 100).toFixed(1)
    : '0';

  return (
    <div className={`border rounded-2xl overflow-hidden transition-all duration-200 ${
      webhook.is_active ? 'border-neutral-200 dark:border-neutral-800' : 'border-neutral-200 dark:border-neutral-800 opacity-60'
    }`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-white/30 dark:hover:bg-neutral-800/30 transition bg-white/40 dark:bg-neutral-900/40"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${
            webhook.is_active ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-400'
          }`}>
            <Webhook className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-xs font-bold text-neutral-900 dark:text-white">{webhook.name}</p>
              {webhook.is_active ? (
                <span className="px-1.5 py-0.5 text-[8px] font-bold uppercase rounded-full bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300">Active</span>
              ) : (
                <span className="px-1.5 py-0.5 text-[8px] font-bold uppercase rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-500">Inactive</span>
              )}
            </div>
            <p className="text-[10px] text-neutral-500 dark:text-neutral-400 font-mono truncate mt-0.5">{webhook.url}</p>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-neutral-400">{successRate}% success</p>
            <p className="text-[10px] text-neutral-500 font-mono">{webhook.failure_count} failures</p>
          </div>
          {expanded ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-neutral-100 dark:border-neutral-800">
          <p className="text-[11px] text-neutral-500 dark:text-neutral-400 mt-3 leading-relaxed">{webhook.description}</p>

          {/* Events */}
          <div className="mt-3">
            <span className="text-[9px] text-neutral-400 uppercase font-semibold">Subscribed Events</span>
            <div className="flex flex-wrap gap-1.5 mt-1">
              {webhook.events.map((ev) => (
                <span key={ev} className={`px-2 py-0.5 text-[9px] font-semibold rounded-full ${EVENT_COLORS[ev] || 'bg-neutral-100 dark:bg-neutral-800 text-neutral-500'}`}>
                  {EVENT_LABELS[ev] || ev}
                </span>
              ))}
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3">
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Success Count</span>
              <p className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold mt-0.5">{webhook.success_count}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Failure Count</span>
              <p className="text-[10px] text-red-600 dark:text-red-400 font-bold mt-0.5">{webhook.failure_count}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Last Status</span>
              <p className={`text-[10px] font-bold mt-0.5 ${webhook.last_status_code && webhook.last_status_code < 400 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}`}>
                {webhook.last_status_code ?? '—'}
              </p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Last Triggered</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 font-mono mt-0.5">
                {webhook.last_triggered_at ? new Date(webhook.last_triggered_at).toLocaleTimeString() : 'Never'}
              </p>
            </div>
          </div>

          {/* Retry Policy */}
          <div className="mt-3 p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
            <span className="text-[9px] text-neutral-400 uppercase font-semibold">Retry Policy</span>
            <div className="flex items-center gap-3 mt-1 text-[10px] text-neutral-600 dark:text-neutral-400">
              <span>Max retries: <strong>{webhook.retry_policy.max_retries}</strong></span>
              <span>Backoff: <strong>{webhook.retry_policy.backoff_ms}ms</strong></span>
              <span>Multiplier: <strong>{webhook.retry_policy.backoff_multiplier}x</strong></span>
            </div>
          </div>

          {/* Headers */}
          <div className="mt-3">
            <span className="text-[9px] text-neutral-400 uppercase font-semibold">Custom Headers</span>
            <div className="flex flex-wrap gap-1.5 mt-1">
              {Object.entries(webhook.headers).map(([k, v]) => (
                <span key={k} className="px-2 py-0.5 bg-neutral-100 dark:bg-neutral-800 text-[9px] text-neutral-600 dark:text-neutral-400 rounded-full font-mono">
                  {k}: {v.length > 20 ? v.substring(0, 20) + '…' : v}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function AlertRuleCard({ rule }: { rule: AlertRule }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`border rounded-2xl overflow-hidden transition-all duration-200 ${
      rule.is_active ? 'border-neutral-200 dark:border-neutral-800' : 'border-neutral-200 dark:border-neutral-800 opacity-60'
    }`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-white/30 dark:hover:bg-neutral-800/30 transition bg-white/40 dark:bg-neutral-900/40"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${
            rule.is_active ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-600 dark:text-amber-400' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-400'
          }`}>
            <Zap className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-bold text-neutral-900 dark:text-white">{rule.name}</p>
            <p className="text-[10px] text-neutral-500 dark:text-neutral-400 truncate">{rule.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] text-neutral-400 font-mono">
            {rule.metric} {rule.operator} {rule.threshold}
          </span>
          {expanded ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-neutral-100 dark:border-neutral-800">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Window</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 mt-0.5">{rule.window_minutes} min</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Cooldown</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 mt-0.5">{rule.cooldown_minutes} min</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Triggered</span>
              <p className="text-[10px] text-amber-600 dark:text-amber-400 font-bold mt-0.5">{rule.trigger_count}x</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Last Triggered</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 font-mono mt-0.5">
                {rule.last_triggered_at ? new Date(rule.last_triggered_at).toLocaleString() : 'Never'}
              </p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Created By</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 mt-0.5">@{rule.created_by}</p>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            <span className="text-[9px] text-neutral-400 uppercase font-semibold">Channels:</span>
            {rule.notification_channels.map((ch) => (
              <span key={ch} className="px-2 py-0.5 bg-neutral-100 dark:bg-neutral-800 text-[9px] text-neutral-500 rounded-full font-medium">
                {ch.replace('_', ' ')}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DeliveryLogRow({ log }: { log: WebhookDeliveryLog }) {
  return (
    <div className="flex items-center gap-3 px-4 py-2.5 border-b border-neutral-100 dark:border-neutral-800 last:border-0 hover:bg-white/20 dark:hover:bg-neutral-800/20 transition">
      <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
        log.success ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600' : 'bg-red-100 dark:bg-red-900/40 text-red-600'
      }`}>
        {log.success ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`px-1.5 py-0.5 text-[8px] font-bold rounded-full ${EVENT_COLORS[log.event_type] || 'bg-neutral-100 text-neutral-500'}`}>
            {EVENT_LABELS[log.event_type] || log.event_type}
          </span>
          <span className={`text-[10px] font-bold ${log.success ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}`}>
            {log.status_code}
          </span>
          <span className="text-[10px] text-neutral-400 font-mono">{log.duration_ms}ms</span>
        </div>
      </div>
      <div className="text-right flex-shrink-0">
        <p className="text-[10px] text-neutral-400 font-mono">{new Date(log.timestamp).toLocaleTimeString()}</p>
        {log.attempt > 1 && <p className="text-[9px] text-amber-500">Attempt {log.attempt}</p>}
      </div>
    </div>
  );
}

export default function WebhookConfigPanel({ webhooks, alertRules, deliveryLogs }: WebhookConfigPanelProps) {
  const [activeTab, setActiveTab] = useState<'webhooks' | 'rules' | 'logs'>('webhooks');

  return (
    <div className="space-y-4">
      {/* Tab Bar */}
      <div className="flex items-center gap-2 p-1 bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-2xl">
        {([
          { key: 'webhooks', label: 'Webhooks', icon: <Webhook className="w-3 h-3" />, count: webhooks.length },
          { key: 'rules', label: 'Alert Rules', icon: <Zap className="w-3 h-3" />, count: alertRules.length },
          { key: 'logs', label: 'Delivery Logs', icon: <Activity className="w-3 h-3" />, count: deliveryLogs.length },
        ] as const).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider rounded-xl transition flex-1 justify-center ${
              activeTab === tab.key
                ? 'bg-amber-500 text-white shadow-lg shadow-amber-500/20'
                : 'text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            }`}
          >
            {tab.icon}
            {tab.label}
            <span className={`px-1.5 py-0.5 text-[8px] font-bold rounded-full ${
              activeTab === tab.key ? 'bg-white/20 text-white' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-500'
            }`}>
              {tab.count}
            </span>
          </button>
        ))}
      </div>

      {/* Webhooks Tab */}
      {activeTab === 'webhooks' && (
        <div className="space-y-3">
          {webhooks.map((wh) => (
            <WebhookCard key={wh.id} webhook={wh} />
          ))}
        </div>
      )}

      {/* Alert Rules Tab */}
      {activeTab === 'rules' && (
        <div className="space-y-3">
          {alertRules.map((rule) => (
            <AlertRuleCard key={rule.id} rule={rule} />
          ))}
        </div>
      )}

      {/* Delivery Logs Tab */}
      {activeTab === 'logs' && (
        <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-neutral-100 dark:border-neutral-800">
            <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Recent Deliveries ({deliveryLogs.length})</span>
          </div>
          {deliveryLogs.map((log) => (
            <DeliveryLogRow key={log.id} log={log} />
          ))}
        </div>
      )}
    </div>
  );
}
