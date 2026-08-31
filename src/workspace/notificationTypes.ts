export type NotificationCategory = 'plagiarism' | 'system' | 'security' | 'compliance' | 'upload' | 'scan' | 'export' | 'user';
export type NotificationPriority = 'low' | 'medium' | 'high' | 'urgent';
export type NotificationChannel = 'in_app' | 'email' | 'slack' | 'webhook' | 'sms';
export type NotificationStatus = 'unread' | 'read' | 'archived' | 'dismissed';
export type WebhookEventType = 'scan.completed' | 'scan.failed' | 'plagiarism.detected' | 'document.uploaded' | 'document.deleted' | 'compliance.breach' | 'threshold.changed' | 'user.role_changed';
export type DigestFrequency = 'realtime' | 'hourly' | 'daily' | 'weekly';
export type AlertRuleOperator = 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'contains';

export interface Notification {
  id: string;
  title: string;
  message: string;
  category: NotificationCategory;
  priority: NotificationPriority;
  channels: NotificationChannel[];
  status: NotificationStatus;
  created_at: string;
  read_at: string | null;
  actor: NotificationActor;
  target: NotificationTarget;
  action_url: string | null;
  metadata: Record<string, string | number | boolean>;
  expires_at: string | null;
  group_id: string | null;
}

export interface NotificationActor {
  user_id: string;
  username: string;
  display_name: string;
  avatar_url?: string;
}

export interface NotificationTarget {
  type: 'document' | 'scan' | 'system' | 'user' | 'report';
  id: string;
  name: string;
}

export interface WebhookConfig {
  id: string;
  name: string;
  url: string;
  secret: string;
  events: WebhookEventType[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_triggered_at: string | null;
  last_status_code: number | null;
  failure_count: number;
  success_count: number;
  headers: Record<string, string>;
  retry_policy: RetryPolicy;
  description: string;
}

export interface RetryPolicy {
  max_retries: number;
  backoff_ms: number;
  backoff_multiplier: number;
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  is_active: boolean;
  metric: string;
  operator: AlertRuleOperator;
  threshold: number;
  window_minutes: number;
  notification_channels: NotificationChannel[];
  cooldown_minutes: number;
  last_triggered_at: string | null;
  trigger_count: number;
  created_at: string;
  created_by: string;
}

export interface DigestConfig {
  id: string;
  user_id: string;
  frequency: DigestFrequency;
  categories: NotificationCategory[];
  quiet_hours_start: string;
  quiet_hours_end: string;
  include_read: boolean;
  max_items_per_digest: number;
  last_sent_at: string | null;
  is_active: boolean;
}

export interface NotificationStats {
  total_notifications: number;
  unread_count: number;
  read_count: number;
  archived_count: number;
  urgent_unread: number;
  today_count: number;
  week_count: number;
  by_category: { category: NotificationCategory; count: number }[];
  by_priority: { priority: NotificationPriority; count: number }[];
  hourly_volume: { hour: number; count: number }[];
  top_actors: { username: string; notification_count: number }[];
}

export interface WebhookDeliveryLog {
  id: string;
  webhook_id: string;
  event_type: WebhookEventType;
  status_code: number;
  success: boolean;
  request_body: string;
  response_body: string;
  duration_ms: number;
  timestamp: string;
  attempt: number;
}

export interface NotificationFilterOptions {
  search: string;
  category: NotificationCategory | '';
  priority: NotificationPriority | '';
  status: NotificationStatus | '';
  channel: NotificationChannel | '';
  date_from: string;
  date_to: string;
  sort_by: 'created_at' | 'priority' | 'category';
  sort_order: 'asc' | 'desc';
  show_archived: boolean;
}
