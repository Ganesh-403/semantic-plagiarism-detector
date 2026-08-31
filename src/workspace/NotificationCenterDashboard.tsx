import React, { useState, useMemo } from 'react';
import {
  Bell,
  BellOff,
  Search,
  Filter,
  RotateCcw,
  CheckCheck,
  Archive,
  Trash2,
  Clock,
  User,
  BarChart3,
  Activity,
  Webhook,
  Zap,
  Settings,
  Mail,
  MessageSquare,
  Smartphone,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  TrendingUp,
  Volume2,
  VolumeX,
  ChevronDown,
  Eye,
  EyeOff,
} from 'lucide-react';
import {
  Notification,
  NotificationFilterOptions,
  NotificationStats,
  NotificationCategory,
  NotificationPriority,
  NotificationStatus,
  WebhookConfig,
  AlertRule,
  WebhookDeliveryLog,
  DigestConfig,
} from './notificationTypes';
import {
  generateMockNotifications,
  generateMockWebhooks,
  generateMockAlertRules,
  generateMockDigestConfig,
  generateMockStats,
  generateMockDeliveryLogs,
} from './notificationMockData';
import NotificationCard from './NotificationCard';
import WebhookConfigPanel from './WebhookConfigPanel';

type ViewMode = 'notifications' | 'webhooks' | 'settings';

function StatCard({ icon, label, value, color = 'bg-amber-50 dark:bg-amber-950/30 text-amber-600 dark:text-amber-400' }: { icon: React.ReactNode; label: string; value: string | number; color?: string }) {
  return (
    <div className="flex items-center gap-3 p-4 bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-2xl hover:shadow-md transition-shadow">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${color}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-lg font-bold text-neutral-900 dark:text-white leading-tight">{value}</p>
        <p className="text-[10px] text-neutral-500 dark:text-neutral-400 font-medium uppercase tracking-wider">{label}</p>
      </div>
    </div>
  );
}

function DigestSettings({ config }: { config: DigestConfig }) {
  const [edited, setEdited] = useState(config);

  return (
    <div className="space-y-4">
      <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-2xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-bold text-neutral-700 dark:text-neutral-300 uppercase tracking-wider">Email Digest Settings</h3>
          <div className="flex items-center gap-2">
            {edited.is_active ? <Volume2 className="w-4 h-4 text-emerald-500" /> : <VolumeX className="w-4 h-4 text-neutral-400" />}
            <button
              onClick={() => setEdited((e) => ({ ...e, is_active: !e.is_active }))}
              className={`relative w-10 h-5 rounded-full transition-colors ${edited.is_active ? 'bg-emerald-500' : 'bg-neutral-300 dark:bg-neutral-700'}`}
            >
              <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${edited.is_active ? 'left-5.5 translate-x-0' : 'left-0.5'}`}
                style={{ left: edited.is_active ? '22px' : '2px' }} />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Frequency */}
          <div>
            <label className="text-[10px] text-neutral-500 uppercase font-semibold mb-1.5 block">Digest Frequency</label>
            <div className="flex gap-2">
              {(['realtime', 'hourly', 'daily', 'weekly'] as const).map((freq) => (
                <button
                  key={freq}
                  onClick={() => setEdited((e) => ({ ...e, frequency: freq }))}
                  className={`flex-1 px-3 py-2 text-[10px] font-semibold rounded-xl transition border ${
                    edited.frequency === freq
                      ? 'bg-amber-500 text-white border-amber-500 shadow-lg shadow-amber-500/20'
                      : 'bg-neutral-50 dark:bg-neutral-950 text-neutral-500 border-neutral-200 dark:border-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                  }`}
                >
                  {freq}
                </button>
              ))}
            </div>
          </div>

          {/* Quiet Hours */}
          <div>
            <label className="text-[10px] text-neutral-500 uppercase font-semibold mb-1.5 block">Quiet Hours</label>
            <div className="flex items-center gap-2">
              <input
                type="time"
                value={edited.quiet_hours_start}
                onChange={(e) => setEdited((ed) => ({ ...ed, quiet_hours_start: e.target.value }))}
                className="flex-1 px-3 py-2 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs font-mono focus:outline-none focus:ring-2 focus:ring-amber-500/20"
              />
              <span className="text-[10px] text-neutral-400">to</span>
              <input
                type="time"
                value={edited.quiet_hours_end}
                onChange={(e) => setEdited((ed) => ({ ...ed, quiet_hours_end: e.target.value }))}
                className="flex-1 px-3 py-2 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs font-mono focus:outline-none focus:ring-2 focus:ring-amber-500/20"
              />
            </div>
          </div>

          {/* Max Items */}
          <div>
            <label className="text-[10px] text-neutral-500 uppercase font-semibold mb-1.5 block">Max Items per Digest</label>
            <input
              type="number"
              value={edited.max_items_per_digest}
              onChange={(e) => setEdited((ed) => ({ ...ed, max_items_per_digest: parseInt(e.target.value) || 25 }))}
              min={5}
              max={100}
              className="w-full px-3 py-2 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs font-mono focus:outline-none focus:ring-2 focus:ring-amber-500/20"
            />
          </div>

          {/* Include Read */}
          <div className="flex items-center gap-3">
            <label className="text-[10px] text-neutral-500 uppercase font-semibold">Include Read Items</label>
            <button
              onClick={() => setEdited((e) => ({ ...e, include_read: !e.include_read }))}
              className={`relative w-10 h-5 rounded-full transition-colors ${edited.include_read ? 'bg-amber-500' : 'bg-neutral-300 dark:bg-neutral-700'}`}
            >
              <div className="absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform"
                style={{ left: edited.include_read ? '22px' : '2px' }} />
            </button>
          </div>
        </div>

        {/* Categories */}
        <div className="mt-4">
          <label className="text-[10px] text-neutral-500 uppercase font-semibold mb-2 block">Included Categories</label>
          <div className="flex flex-wrap gap-2">
            {(['plagiarism', 'system', 'security', 'compliance', 'upload', 'scan', 'export', 'user'] as NotificationCategory[]).map((cat) => {
              const isActive = edited.categories.includes(cat);
              return (
                <button
                  key={cat}
                  onClick={() => {
                    setEdited((e) => ({
                      ...e,
                      categories: isActive ? e.categories.filter((c) => c !== cat) : [...e.categories, cat],
                    }));
                  }}
                  className={`px-3 py-1.5 text-[10px] font-semibold rounded-xl transition border ${
                    isActive
                      ? 'bg-amber-500 text-white border-amber-500'
                      : 'bg-neutral-50 dark:bg-neutral-950 text-neutral-500 border-neutral-200 dark:border-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                  }`}
                >
                  {cat}
                </button>
              );
            })}
          </div>
        </div>

        {/* Last Sent */}
        <div className="mt-4 flex items-center gap-4 text-[10px] text-neutral-400">
          <span>Last sent: {edited.last_sent_at ? new Date(edited.last_sent_at).toLocaleString() : 'Never'}</span>
          <span>•</span>
          <span>Digest ID: {edited.id}</span>
        </div>
      </div>
    </div>
  );
}

function ChannelDistributionBar({ stats }: { stats: NotificationStats }) {
  const total = stats.by_category.reduce((sum, c) => sum + c.count, 0);
  const categoryColors: Record<string, string> = {
    plagiarism: 'bg-red-500',
    scan: 'bg-violet-500',
    upload: 'bg-emerald-500',
    system: 'bg-neutral-500',
    security: 'bg-orange-500',
    compliance: 'bg-amber-500',
    export: 'bg-blue-500',
    user: 'bg-pink-500',
  };

  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-2xl p-4">
      <h4 className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider mb-3">Notifications by Category</h4>
      <div className="w-full h-3 bg-neutral-100 dark:bg-neutral-800 rounded-full overflow-hidden flex">
        {stats.by_category.map((cat) => (
          <div
            key={cat.category}
            className={`${categoryColors[cat.category] || 'bg-neutral-400'} h-full transition-all duration-500`}
            style={{ width: `${(cat.count / total) * 100}%` }}
            title={`${cat.category}: ${cat.count}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-3 mt-2">
        {stats.by_category.map((cat) => (
          <span key={cat.category} className="flex items-center gap-1.5 text-[9px] text-neutral-500">
            <span className={`w-2 h-2 rounded-full ${categoryColors[cat.category] || 'bg-neutral-400'}`} />
            {cat.category} ({cat.count})
          </span>
        ))}
      </div>
    </div>
  );
}

export default function NotificationCenterDashboard() {
  const allNotifications = useMemo(() => generateMockNotifications(), []);
  const webhooks = useMemo(() => generateMockWebhooks(), []);
  const alertRules = useMemo(() => generateMockAlertRules(), []);
  const digestConfig = useMemo(() => generateMockDigestConfig(), []);
  const stats = useMemo(() => generateMockStats(), []);
  const deliveryLogs = useMemo(() => generateMockDeliveryLogs(), []);

  const [viewMode, setViewMode] = useState<ViewMode>('notifications');
  const [notifications, setNotifications] = useState(allNotifications);
  const [filters, setFilters] = useState<NotificationFilterOptions>({
    search: '',
    category: '',
    priority: '',
    status: '',
    channel: '',
    date_from: '',
    date_to: '',
    sort_by: 'created_at',
    sort_order: 'desc',
    show_archived: false,
  });

  const filteredNotifications = useMemo(() => {
    return notifications.filter((n) => {
      if (filters.search && !n.title.toLowerCase().includes(filters.search.toLowerCase()) && !n.message.toLowerCase().includes(filters.search.toLowerCase())) return false;
      if (filters.category && n.category !== filters.category) return false;
      if (filters.priority && n.priority !== filters.priority) return false;
      if (filters.status && n.status !== filters.status) return false;
      if (filters.channel && !n.channels.includes(filters.channel as any)) return false;
      if (!filters.show_archived && n.status === 'archived') return false;
      return true;
    }).sort((a, b) => {
      const cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      return filters.sort_order === 'desc' ? -cmp : cmp;
    });
  }, [notifications, filters]);

  const markRead = (id: string) => setNotifications((prev) => prev.map((n) => n.id === id ? { ...n, status: 'read' as const, read_at: new Date().toISOString() } : n));
  const archive = (id: string) => setNotifications((prev) => prev.map((n) => n.id === id ? { ...n, status: 'archived' as const } : n));
  const dismiss = (id: string) => setNotifications((prev) => prev.filter((n) => n.id !== id));
  const markAllRead = () => setNotifications((prev) => prev.map((n) => n.status === 'unread' ? { ...n, status: 'read' as const, read_at: new Date().toISOString() } : n));

  const resetFilters = () => setFilters({ search: '', category: '', priority: '', status: '', channel: '', date_from: '', date_to: '', sort_by: 'created_at', sort_order: 'desc', show_archived: false });

  const unreadCount = notifications.filter((n) => n.status === 'unread').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 via-white to-amber-50/30 dark:from-neutral-950 dark:via-neutral-900 dark:to-amber-950/10 p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Title Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold text-neutral-900 dark:text-white flex items-center gap-2">
              <Bell className="w-5 h-5 text-amber-500" />
              Notification Center
              {unreadCount > 0 && (
                <span className="px-2 py-0.5 bg-red-500 text-white text-[10px] font-bold rounded-full animate-pulse">
                  {unreadCount}
                </span>
              )}
            </h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
              Manage notifications, webhooks, alert rules, and digest preferences
            </p>
          </div>
          <div className="flex items-center gap-2">
            {unreadCount > 0 && (
              <button onClick={markAllRead} className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-semibold bg-emerald-500 text-white rounded-xl hover:bg-emerald-600 transition shadow-lg shadow-emerald-500/20">
                <CheckCheck className="w-3 h-3" />
                Mark All Read
              </button>
            )}
            {(['notifications', 'webhooks', 'settings'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider rounded-xl transition ${
                  viewMode === mode
                    ? 'bg-amber-500 text-white shadow-lg shadow-amber-500/20'
                    : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
                }`}
              >
                {mode === 'notifications' && <Bell className="w-3 h-3 inline mr-1" />}
                {mode === 'webhooks' && <Webhook className="w-3 h-3 inline mr-1" />}
                {mode === 'settings' && <Settings className="w-3 h-3 inline mr-1" />}
                {mode}
              </button>
            ))}
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard icon={<Bell className="w-5 h-5" />} label="Unread" value={stats.unread_count} color={stats.urgent_unread > 0 ? 'bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400' : undefined} />
          <StatCard icon={<Clock className="w-5 h-5" />} label="Today" value={stats.today_count} />
          <StatCard icon={<Activity className="w-5 h-5" />} label="This Week" value={stats.week_count} color="bg-blue-50 dark:bg-blue-950/30 text-blue-600 dark:text-blue-400" />
          <StatCard icon={<AlertTriangle className="w-5 h-5" />} label="Urgent Unread" value={stats.urgent_unread} color="bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400" />
        </div>

        {/* Category Distribution */}
        {viewMode === 'notifications' && <ChannelDistributionBar stats={stats} />}

        {/* View: Notifications */}
        {viewMode === 'notifications' && (
          <div className="space-y-4">
            {/* Filter Bar */}
            <div className="flex flex-col lg:flex-row gap-3 p-4 bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-2xl">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
                <input
                  type="text"
                  placeholder="Search notifications..."
                  value={filters.search}
                  onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
                  className="w-full pl-10 pr-4 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 transition"
                />
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <select value={filters.category} onChange={(e) => setFilters((f) => ({ ...f, category: e.target.value as NotificationCategory | '' }))} className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium">
                  <option value="">All Categories</option>
                  {(['plagiarism', 'scan', 'upload', 'system', 'security', 'compliance', 'export', 'user'] as const).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <select value={filters.priority} onChange={(e) => setFilters((f) => ({ ...f, priority: e.target.value as NotificationPriority | '' }))} className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium">
                  <option value="">All Priorities</option>
                  <option value="urgent">Urgent</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value as NotificationStatus | '' }))} className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium">
                  <option value="">All Statuses</option>
                  <option value="unread">Unread</option>
                  <option value="read">Read</option>
                  <option value="archived">Archived</option>
                </select>
                <label className="flex items-center gap-1.5 px-3 py-2 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-[10px] font-medium cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-800 transition">
                  <input type="checkbox" checked={filters.show_archived} onChange={(e) => setFilters((f) => ({ ...f, show_archived: e.target.checked }))} className="rounded border-neutral-300 text-amber-500 focus:ring-amber-500/20 w-3 h-3 accent-amber-500" />
                  Show Archived
                </label>
                <button onClick={resetFilters} className="p-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl hover:bg-neutral-100 dark:hover:bg-neutral-800 transition" title="Reset">
                  <RotateCcw className="w-3.5 h-3.5 text-neutral-500" />
                </button>
              </div>
            </div>

            {/* Notification List */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-neutral-500">{filteredNotifications.length} notifications</span>
              </div>
              {filteredNotifications.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-neutral-400">
                  <BellOff className="w-8 h-8 mb-2" />
                  <p className="text-xs font-medium">No notifications match the current filters.</p>
                </div>
              ) : (
                filteredNotifications.map((n) => (
                  <NotificationCard key={n.id} notification={n} onMarkRead={markRead} onArchive={archive} onDismiss={dismiss} />
                ))
              )}
            </div>
          </div>
        )}

        {/* View: Webhooks */}
        {viewMode === 'webhooks' && (
          <WebhookConfigPanel webhooks={webhooks} alertRules={alertRules} deliveryLogs={deliveryLogs} />
        )}

        {/* View: Settings */}
        {viewMode === 'settings' && (
          <DigestSettings config={digestConfig} />
        )}
      </div>
    </div>
  );
}
