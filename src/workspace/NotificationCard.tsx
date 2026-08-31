import React, { useState } from 'react';
import {
  Bell,
  BellOff,
  Check,
  Archive,
  Trash2,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Clock,
  User,
  AlertTriangle,
  AlertOctagon,
  Info,
  Shield,
  Upload,
  Activity,
  FileText,
  Settings,
  Mail,
  MessageSquare,
  Smartphone,
  Tag,
} from 'lucide-react';
import { Notification } from './notificationTypes';

interface NotificationCardProps {
  notification: Notification;
  onMarkRead: (id: string) => void;
  onArchive: (id: string) => void;
  onDismiss: (id: string) => void;
}

const priorityConfig: Record<string, { icon: React.ReactNode; color: string; bg: string; border: string; pulse?: boolean }> = {
  urgent: { icon: <AlertOctagon className="w-3.5 h-3.5" />, color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/40', border: 'border-red-300 dark:border-red-700', pulse: true },
  high: { icon: <AlertTriangle className="w-3.5 h-3.5" />, color: 'text-orange-600 dark:text-orange-400', bg: 'bg-orange-100 dark:bg-orange-900/40', border: 'border-orange-300 dark:border-orange-700' },
  medium: { icon: <Info className="w-3.5 h-3.5" />, color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-100 dark:bg-amber-900/40', border: 'border-amber-300 dark:border-amber-700' },
  low: { icon: <Info className="w-3.5 h-3.5" />, color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-100 dark:bg-blue-900/40', border: 'border-blue-300 dark:border-blue-700' },
};

const categoryConfig: Record<string, { icon: React.ReactNode; label: string; color: string }> = {
  plagiarism: { icon: <Shield className="w-3 h-3" />, label: 'Plagiarism', color: 'text-red-600 dark:text-red-400' },
  scan: { icon: <Activity className="w-3 h-3" />, label: 'Scan', color: 'text-violet-600 dark:text-violet-400' },
  upload: { icon: <Upload className="w-3 h-3" />, label: 'Upload', color: 'text-emerald-600 dark:text-emerald-400' },
  system: { icon: <Settings className="w-3 h-3" />, label: 'System', color: 'text-neutral-600 dark:text-neutral-400' },
  security: { icon: <AlertOctagon className="w-3 h-3" />, label: 'Security', color: 'text-red-600 dark:text-red-400' },
  compliance: { icon: <FileText className="w-3 h-3" />, label: 'Compliance', color: 'text-amber-600 dark:text-amber-400' },
  export: { icon: <FileText className="w-3 h-3" />, label: 'Export', color: 'text-blue-600 dark:text-blue-400' },
  user: { icon: <User className="w-3 h-3" />, label: 'User', color: 'text-pink-600 dark:text-pink-400' },
};

const channelIcons: Record<string, React.ReactNode> = {
  in_app: <Bell className="w-3 h-3" />,
  email: <Mail className="w-3 h-3" />,
  slack: <MessageSquare className="w-3 h-3" />,
  webhook: <ExternalLink className="w-3 h-3" />,
  sms: <Smartphone className="w-3 h-3" />,
};

function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const then = new Date(timestamp);
  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return then.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export default function NotificationCard({ notification, onMarkRead, onArchive, onDismiss }: NotificationCardProps) {
  const [expanded, setExpanded] = useState(false);
  const pConfig = priorityConfig[notification.priority] || priorityConfig.low;
  const cConfig = categoryConfig[notification.category] || categoryConfig.system;
  const isUnread = notification.status === 'unread';
  const isArchived = notification.status === 'archived';

  return (
    <div className={`border rounded-2xl overflow-hidden transition-all duration-200 ${
      isUnread ? `${pConfig.border} border-2 shadow-sm` : 'border-neutral-200 dark:border-neutral-800'
    } ${isArchived ? 'opacity-50' : ''} ${isUnread ? 'bg-white/80 dark:bg-neutral-900/80' : 'bg-white/40 dark:bg-neutral-900/40'} backdrop-blur-md`}>
      <div className="flex items-start gap-3 px-4 py-3">
        {/* Priority Icon */}
        <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${pConfig.bg} ${pConfig.color} flex-shrink-0 mt-0.5 ${pConfig.pulse ? 'animate-pulse' : ''}`}>
          {pConfig.icon}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className={`px-2 py-0.5 text-[9px] font-bold uppercase rounded-full ${pConfig.bg} ${pConfig.color}`}>
              {notification.priority}
            </span>
            <span className={`flex items-center gap-1 text-[10px] font-semibold ${cConfig.color}`}>
              {cConfig.icon}
              {cConfig.label}
            </span>
            {isUnread && (
              <span className="w-2 h-2 rounded-full bg-amber-500 flex-shrink-0" />
            )}
          </div>
          <p className={`text-xs font-bold leading-tight ${isUnread ? 'text-neutral-900 dark:text-white' : 'text-neutral-700 dark:text-neutral-300'}`}>
            {notification.title}
          </p>
          <p className="text-[11px] text-neutral-500 dark:text-neutral-400 leading-relaxed mt-0.5 line-clamp-2">
            {notification.message}
          </p>
          <div className="flex items-center gap-3 mt-1.5 text-[10px] text-neutral-400 dark:text-neutral-500">
            <span className="flex items-center gap-1">
              <User className="w-3 h-3" />
              {notification.actor.display_name}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatRelativeTime(notification.created_at)}
            </span>
            <div className="flex items-center gap-1">
              {notification.channels.map((ch) => (
                <span key={ch} className="text-neutral-300 dark:text-neutral-600" title={ch}>
                  {channelIcons[ch] || <Bell className="w-3 h-3" />}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {isUnread && (
            <button onClick={() => onMarkRead(notification.id)} className="p-1.5 rounded-lg hover:bg-emerald-50 dark:hover:bg-emerald-950/30 text-neutral-400 hover:text-emerald-500 transition" title="Mark as read">
              <Check className="w-3.5 h-3.5" />
            </button>
          )}
          {!isArchived && (
            <button onClick={() => onArchive(notification.id)} className="p-1.5 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-950/30 text-neutral-400 hover:text-blue-500 transition" title="Archive">
              <Archive className="w-3.5 h-3.5" />
            </button>
          )}
          <button onClick={() => onDismiss(notification.id)} className="p-1.5 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 text-neutral-400 hover:text-red-500 transition" title="Dismiss">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setExpanded(!expanded)} className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-400 transition">
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-neutral-100 dark:border-neutral-800">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Notification ID</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 font-mono mt-0.5 truncate">{notification.id}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Created</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 font-mono mt-0.5">{new Date(notification.created_at).toLocaleString()}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Status</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 mt-0.5 capitalize">{notification.status}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Actor</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 mt-0.5">@{notification.actor.username}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Target</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 mt-0.5 truncate">{notification.target.name}</p>
            </div>
            <div className="p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Expires</span>
              <p className="text-[10px] text-neutral-700 dark:text-neutral-300 font-mono mt-0.5">
                {notification.expires_at ? new Date(notification.expires_at).toLocaleDateString() : 'Never'}
              </p>
            </div>
          </div>

          {/* Channels */}
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            <span className="text-[9px] text-neutral-400 uppercase font-semibold">Channels:</span>
            {notification.channels.map((ch) => (
              <span key={ch} className="flex items-center gap-1 px-2 py-0.5 bg-neutral-100 dark:bg-neutral-800 text-[9px] text-neutral-600 dark:text-neutral-400 rounded-full font-medium">
                {channelIcons[ch]}
                {ch.replace('_', ' ')}
              </span>
            ))}
          </div>

          {/* Metadata */}
          {Object.keys(notification.metadata).length > 0 && (
            <div className="mt-3 p-2 bg-neutral-50 dark:bg-neutral-950 rounded-xl">
              <span className="text-[9px] text-neutral-400 uppercase font-semibold">Metadata</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {Object.entries(notification.metadata).map(([k, v]) => (
                  <span key={k} className="px-2 py-0.5 bg-neutral-100 dark:bg-neutral-800 text-[9px] text-neutral-600 dark:text-neutral-400 rounded-full font-mono">
                    {k}: {String(v)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Action URL */}
          {notification.action_url && (
            <div className="mt-3">
              <a href={notification.action_url} className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 text-[10px] font-semibold rounded-xl hover:bg-amber-100 dark:hover:bg-amber-900/50 transition border border-amber-200 dark:border-amber-800">
                <ExternalLink className="w-3 h-3" />
                View Details
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
