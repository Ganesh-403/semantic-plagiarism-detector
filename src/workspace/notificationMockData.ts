import {
  Notification,
  WebhookConfig,
  AlertRule,
  DigestConfig,
  NotificationStats,
  WebhookDeliveryLog,
  NotificationCategory,
  NotificationPriority,
  NotificationStatus,
  WebhookEventType,
} from './notificationTypes';

function rid(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).substring(2, 10)}`;
}

const ACTORS = [
  { user_id: 'u_001', username: 'prof_jackson', display_name: 'Prof. Jackson' },
  { user_id: 'u_002', username: 'dr_dupont', display_name: 'Dr. Dupont' },
  { user_id: 'u_003', username: 'alex_rivera', display_name: 'Alex Rivera' },
  { user_id: 'u_004', username: 'admin_sys', display_name: 'System Admin' },
  { user_id: 'u_005', username: 'beatrix_v', display_name: 'Beatrix Vance' },
  { user_id: 'u_006', username: 'chloe_l', display_name: 'Chloe Laurent' },
];

const NOTIFICATION_TEMPLATES: { category: NotificationCategory; title: string; message: string; priority: NotificationPriority }[] = [
  { category: 'plagiarism', title: 'High Similarity Detected', message: 'Document "neural_attention_analysis.pdf" scored 87% similarity with "transformer_encoding_v3.pdf". Review recommended.', priority: 'high' },
  { category: 'plagiarism', title: 'Plagiarism Cluster Found', message: '3 documents in CS-401 share >80% overlap. Cluster includes: paper1.pdf, paper2.pdf, paper3.pdf.', priority: 'urgent' },
  { category: 'scan', title: 'Batch Scan Completed', message: '14 documents processed in batch scan. 2 flagged for review, 12 passed with <15% similarity.', priority: 'medium' },
  { category: 'scan', title: 'Scan Failed — Encoding Error', message: 'Document "corrupted_upload.zip" failed scan due to unsupported file encoding. Manual intervention required.', priority: 'high' },
  { category: 'upload', title: 'New Document Uploaded', message: 'Alex Rivera uploaded "positional_encoding_benchmarks.pdf" to CS-401 corpus. Auto-scan queued.', priority: 'low' },
  { category: 'upload', title: 'Bulk Upload Complete', message: 'Beatrix Vance uploaded 8 documents in bulk. All files indexed and chunked successfully.', priority: 'medium' },
  { category: 'system', title: 'FAISS Index Rebuilt', message: 'Vector index rebuilt in 4.2s. 42 documents, 382 chunks indexed. Memory usage: 67%.', priority: 'low' },
  { category: 'system', title: 'Storage Threshold Warning', message: 'Corpus storage at 78% capacity (31.2GB / 40GB). Consider archiving old documents.', priority: 'medium' },
  { category: 'security', title: 'Brute Force Login Detected', message: '5 failed login attempts from IP 203.0.113.5 targeting account prof_jackson within 2 minutes.', priority: 'urgent' },
  { category: 'security', title: 'API Key Expiring Soon', message: 'Production API key expires in 3 days. Rotate key to avoid service disruption.', priority: 'high' },
  { category: 'compliance', title: 'Compliance Check Failed', message: '3 documents exceed 365-day retention policy. Action required to maintain compliance.', priority: 'high' },
  { category: 'compliance', title: 'Scan Coverage Below Threshold', message: 'Only 94% of documents scanned within 24h SLA. 3 documents pending scan.', priority: 'medium' },
  { category: 'export', title: 'Report Generated', message: 'Monthly plagiarism report for August 2026 is ready for download. 42 documents analyzed.', priority: 'low' },
  { category: 'export', title: 'Bulk Export Failed', message: 'Export of 50 documents timed out after 120s. Partial export available (32/50 files).', priority: 'high' },
  { category: 'user', title: 'New User Registered', message: 'Fujita Sato joined as a viewer in CS-502. Pending admin approval for document access.', priority: 'low' },
  { category: 'user', title: 'Role Changed', message: 'Chloe Laurent promoted from viewer to editor by Prof. Jackson.', priority: 'medium' },
];

function randomNotification(hoursAgo: number): Notification {
  const template = NOTIFICATION_TEMPLATES[Math.floor(Math.random() * NOTIFICATION_TEMPLATES.length)];
  const actor = ACTORS[Math.floor(Math.random() * ACTORS.length)];
  const statuses: NotificationStatus[] = ['unread', 'read', 'archived', 'dismissed'];
  const status = statuses[Math.floor(Math.random() * statuses.length)];

  return {
    id: rid('notif'),
    title: template.title,
    message: template.message,
    category: template.category,
    priority: template.priority,
    channels: ['in_app', ...(Math.random() > 0.5 ? ['email'] : []), ...(Math.random() > 0.8 ? ['slack'] : [])] as any[],
    status,
    created_at: new Date(Date.now() - hoursAgo * 3600000).toISOString(),
    read_at: status !== 'unread' ? new Date(Date.now() - (hoursAgo - 1) * 3600000).toISOString() : null,
    actor: { ...actor },
    target: { type: 'document', id: rid('doc'), name: 'sample_document.pdf' },
    action_url: `/documents/${rid('doc')}`,
    metadata: { scan_id: rid('scan'), similarity_score: Math.floor(Math.random() * 40) + 60 },
    expires_at: new Date(Date.now() + 7 * 86400000).toISOString(),
    group_id: Math.random() > 0.7 ? `grp_${Math.floor(Math.random() * 5)}` : null,
  };
}

export function generateMockNotifications(): Notification[] {
  const notifications: Notification[] = [];
  for (let i = 0; i < 40; i++) {
    const hoursAgo = Math.floor(Math.random() * 168);
    notifications.push(randomNotification(hoursAgo));
  }
  return notifications.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
}

export function generateMockWebhooks(): WebhookConfig[] {
  return [
    { id: rid('wh'), name: 'Slack Alert Bot', url: 'https://hooks.slack.com/services/T0X/B0Y/abc123', secret: 'whsec_****7f3a', events: ['plagiarism.detected', 'scan.failed', 'compliance.breach'], is_active: true, created_at: new Date(Date.now() - 30 * 86400000).toISOString(), updated_at: new Date(Date.now() - 2 * 86400000).toISOString(), last_triggered_at: new Date(Date.now() - 3600000).toISOString(), last_status_code: 200, failure_count: 2, success_count: 147, headers: { 'Content-Type': 'application/json' }, retry_policy: { max_retries: 3, backoff_ms: 1000, backoff_multiplier: 2 }, description: 'Posts plagiarism alerts to #academic-integrity channel' },
    { id: rid('wh'), name: 'Analytics Pipeline', url: 'https://analytics.uni.edu/api/ingest', secret: 'whsec_****9b2c', events: ['scan.completed', 'document.uploaded', 'document.deleted'], is_active: true, created_at: new Date(Date.now() - 60 * 86400000).toISOString(), updated_at: new Date(Date.now() - 5 * 86400000).toISOString(), last_triggered_at: new Date(Date.now() - 7200000).toISOString(), last_status_code: 201, failure_count: 8, success_count: 312, headers: { 'Content-Type': 'application/json', 'X-API-Key': 'analytics_key_****' }, retry_policy: { max_retries: 5, backoff_ms: 2000, backoff_multiplier: 2 }, description: 'Sends scan and document events to university analytics pipeline' },
    { id: rid('wh'), name: 'Compliance Notifier', url: 'https://compliance.uni.edu/webhooks/audit', secret: 'whsec_****4d1e', events: ['compliance.breach', 'threshold.changed', 'user.role_changed'], is_active: true, created_at: new Date(Date.now() - 45 * 86400000).toISOString(), updated_at: new Date(Date.now() - 1 * 86400000).toISOString(), last_triggered_at: new Date(Date.now() - 14400000).toISOString(), last_status_code: 200, failure_count: 0, success_count: 23, headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer comp_token_****' }, retry_policy: { max_retries: 3, backoff_ms: 5000, backoff_multiplier: 3 }, description: 'Notifies compliance team of policy breaches and config changes' },
    { id: rid('wh'), name: 'Backup Trigger', url: 'https://backup.uni.edu/api/trigger', secret: 'whsec_****6a8f', events: ['document.deleted', 'scan.completed'], is_active: false, created_at: new Date(Date.now() - 90 * 86400000).toISOString(), updated_at: new Date(Date.now() - 30 * 86400000).toISOString(), last_triggered_at: new Date(Date.now() - 7 * 86400000).toISOString(), last_status_code: 503, failure_count: 15, success_count: 89, headers: { 'Content-Type': 'application/json' }, retry_policy: { max_retries: 2, backoff_ms: 10000, backoff_multiplier: 1 }, description: 'Triggers incremental backup after document changes (currently disabled)' },
  ];
}

export function generateMockAlertRules(): AlertRule[] {
  return [
    { id: rid('rule'), name: 'High Plagiarism Score', description: 'Alert when any document similarity score exceeds 85%', is_active: true, metric: 'similarity_score', operator: 'gt', threshold: 85, window_minutes: 60, notification_channels: ['in_app', 'email', 'slack'], cooldown_minutes: 30, last_triggered_at: new Date(Date.now() - 7200000).toISOString(), trigger_count: 12, created_at: new Date(Date.now() - 60 * 86400000).toISOString(), created_by: 'prof_jackson' },
    { id: rid('rule'), name: 'Scan Failure Spike', description: 'Alert when more than 3 scans fail in a 1-hour window', is_active: true, metric: 'scan_failures', operator: 'gt', threshold: 3, window_minutes: 60, notification_channels: ['in_app', 'slack'], cooldown_minutes: 15, last_triggered_at: new Date(Date.now() - 86400000).toISOString(), trigger_count: 4, created_at: new Date(Date.now() - 30 * 86400000).toISOString(), created_by: 'admin_sys' },
    { id: rid('rule'), name: 'Login Brute Force', description: 'Alert when 5+ failed logins from same IP within 5 minutes', is_active: true, metric: 'failed_logins', operator: 'gte', threshold: 5, window_minutes: 5, notification_channels: ['in_app', 'email', 'sms'], cooldown_minutes: 10, last_triggered_at: new Date(Date.now() - 3600000).toISOString(), trigger_count: 3, created_at: new Date(Date.now() - 90 * 86400000).toISOString(), created_by: 'admin_sys' },
    { id: rid('rule'), name: 'Storage Capacity Warning', description: 'Alert when corpus storage exceeds 75% capacity', is_active: true, metric: 'storage_percent', operator: 'gt', threshold: 75, window_minutes: 1440, notification_channels: ['in_app', 'email'], cooldown_minutes: 1440, last_triggered_at: new Date(Date.now() - 43200000).toISOString(), trigger_count: 6, created_at: new Date(Date.now() - 120 * 86400000).toISOString(), created_by: 'admin_sys' },
    { id: rid('rule'), name: 'Bulk Upload Alert', description: 'Alert when single user uploads more than 10 documents in 1 hour', is_active: false, metric: 'upload_count', operator: 'gt', threshold: 10, window_minutes: 60, notification_channels: ['in_app'], cooldown_minutes: 60, last_triggered_at: null, trigger_count: 0, created_at: new Date(Date.now() - 15 * 86400000).toISOString(), created_by: 'dr_dupont' },
  ];
}

export function generateMockDigestConfig(): DigestConfig {
  return {
    id: rid('digest'),
    user_id: 'u_001',
    frequency: 'daily',
    categories: ['plagiarism', 'security', 'compliance', 'scan'],
    quiet_hours_start: '22:00',
    quiet_hours_end: '07:00',
    include_read: false,
    max_items_per_digest: 25,
    last_sent_at: new Date(Date.now() - 12 * 3600000).toISOString(),
    is_active: true,
  };
}

export function generateMockStats(): NotificationStats {
  return {
    total_notifications: 156,
    unread_count: 12,
    read_count: 98,
    archived_count: 34,
    urgent_unread: 2,
    today_count: 8,
    week_count: 42,
    by_category: [
      { category: 'plagiarism', count: 38 },
      { category: 'scan', count: 32 },
      { category: 'upload', count: 28 },
      { category: 'system', count: 22 },
      { category: 'security', count: 18 },
      { category: 'compliance', count: 12 },
      { category: 'export', count: 4 },
      { category: 'user', count: 2 },
    ],
    by_priority: [
      { priority: 'urgent', count: 8 },
      { priority: 'high', count: 24 },
      { priority: 'medium', count: 52 },
      { priority: 'low', count: 72 },
    ],
    hourly_volume: Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      count: Math.floor(Math.random() * 15) + (i >= 8 && i <= 18 ? 5 : 1),
    })),
    top_actors: [
      { username: 'prof_jackson', notification_count: 35 },
      { username: 'admin_sys', notification_count: 28 },
      { username: 'dr_dupont', notification_count: 22 },
      { username: 'alex_rivera', notification_count: 18 },
    ],
  };
}

export function generateMockDeliveryLogs(): WebhookDeliveryLog[] {
  const events: WebhookEventType[] = ['scan.completed', 'plagiarism.detected', 'document.uploaded', 'compliance.breach'];
  const logs: WebhookDeliveryLog[] = [];

  for (let i = 0; i < 15; i++) {
    const success = Math.random() > 0.2;
    logs.push({
      id: rid('log'),
      webhook_id: rid('wh'),
      event_type: events[Math.floor(Math.random() * events.length)],
      status_code: success ? 200 : [400, 403, 500, 503][Math.floor(Math.random() * 4)],
      success,
      request_body: JSON.stringify({ event: 'scan.completed', document_id: rid('doc') }),
      response_body: success ? '{"ok":true}' : '{"error":"internal server error"}',
      duration_ms: Math.floor(Math.random() * 2000) + 50,
      timestamp: new Date(Date.now() - Math.floor(Math.random() * 86400000)).toISOString(),
      attempt: success ? 1 : Math.floor(Math.random() * 3) + 1,
    });
  }

  return logs.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}
