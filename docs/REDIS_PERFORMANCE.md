# Redis Performance Tuning Guide

This guide provides comprehensive instructions and best practices for configuring, optimizing, and maintaining Redis as a high-performance caching layer in a production environment.

In the **Semantic Plagiarism Detector**, Redis acts as a critical speed-up mechanism and state manager. It handles:

* **Session Caching**: Serialized user session states (`spd:v1:session:<id>:<key>`) with a short Time-To-Live (TTL) of 15 minutes.
* **FAISS Index Caching**: Heavy binary representations of vector similarity search indexes (`spd:v1:faiss:index:<key>`) cached for 24 hours.
* **Analysis Results Caching**: Document analysis results and embeddings (`spd:v1:analysis:<key>`) cached for 2 hours.
* **Security & Rate Limiting**: Counter keys for login lockout tracking (`spd:v1:login_attempts:<id>`) and upload limits (`spd:v1:uploads:<username>`).

Proper Redis tuning ensures system responsiveness, prevents Out-Of-Memory (OOM) errors, and optimizes resource consumption under high concurrent workloads.

---

## 1. Redis Memory Settings

Redis holds its entire dataset in RAM to deliver sub-millisecond response times. Therefore, precise memory allocation is crucial to prevent system instability, memory fragmentation, and swapping to disk.

### 1.1 Memory Limit (`maxmemory`)

By default, on 64-bit systems, Redis has no memory limit and will continue consuming RAM until the host system runs out of memory. This triggers the operating system's OOM Killer to terminate processes (often Redis itself).

In production, you should set a strict upper bound using the `maxmemory` setting.

#### Recommended Allocation Rules

* **Dedicated Redis Host**: Allocate **60% to 70%** of total system RAM to Redis, leaving the rest for the operating system, network buffers, and persistence overhead (such as process forking).
* **Shared Host (e.g., App + Redis on same VM)**: Limit Redis to **25% to 30%** of total system RAM or a fixed value (e.g., `2GB`), ensuring it does not starve CPU-intensive vector/embedding operations.

#### Setting `maxmemory` in `redis.conf`

```ini
# Limit Redis memory consumption to 2 Gigabytes
maxmemory 2gb
```

#### Setting `maxmemory` dynamically (without restarting Redis)

```bash
redis-cli CONFIG SET maxmemory 2gb
```

---

## 2. Eviction Policies

When the memory usage reaches the defined `maxmemory` threshold, Redis must decide how to handle new write requests. This is governed by the `maxmemory-policy`.

### 2.1 Why `allkeys-lru` is Highly Recommended

For the Semantic Plagiarism Detector, the **`allkeys-lru`** eviction policy is the recommended default.

* **Reclaims Space Automatically**: If memory becomes full due to large FAISS vector indexes or active sessions, `allkeys-lru` will evict the **Least Recently Used (LRU)** keys across the entire database, regardless of whether they have a set expiration (TTL).
* **Guarantees Availability**: Under heavy load, instead of throwing write errors, Redis silently discards older or inactive cache data (such as historical similarity matrices or closed session data) to make room for active calculations.
* **Preserves Hot Data**: High-frequency items (like actively queried vector databases or current user sessions) are preserved since their access frequency keeps them "recently used".

### 2.2 Eviction Policies Matrix

| Policy Name | Description | Best Used For |
| :--- | :--- | :--- |
| **`allkeys-lru`** | Evicts the least recently used keys across the entire keyspace. | **Standard Caching (Recommended)**. Prevents OOM by sacrificing old/unused cache items. |
| **`volatile-lru`** | Evicts the least recently used keys, but only those with an expiration (`TTL`) set. | Hybrid usage where some keys are persistent (no TTL) and others are transient caches. |
| **`allkeys-lfu`** | Evicts the least frequently used keys (least accessed count) across the entire keyspace. | Caching patterns where access frequency is a better metric of value than recency. |
| **`volatile-lfu`** | Evicts the least frequently used keys, but only those with an expiration set. | Frequency-based eviction targeting transient data. |
| **`allkeys-random`** | Evicts random keys across the entire keyspace. | Rarely used; only suitable if all keys have equal value and access patterns. |
| **`volatile-random`** | Evicts random keys, but only those with an expiration set. | Rarely used; non-deterministic eviction of transient keys. |
| **`volatile-ttl`** | Evicts keys with an expiration set, prioritizing those with the shortest remaining TTL. | Applications where expiring data must be cleared as early as possible under memory pressure. |
| **`noeviction`** | Never evicts keys. Returns an out-of-memory error `(error) OOM command not allowed` for writes. | **Strictly Data-Store / No Data Loss**. Used when Redis functions as a database or message broker. |

---

## 3. Persistence Options

Redis offers two main persistence mechanisms to save data to disk: **RDB (Redis Database snapshots)** and **AOF (Append Only File)**. Understanding their differences is key to optimizing performance.

### 3.1 RDB (Redis Database Snapshotting)

RDB persistence performs point-in-time snapshots of the dataset at specified intervals.

* **How it works**: Redis forks a child process. The child writes the memory snapshot to a temporary RDB file (`dump.rdb`) and replaces the old file when done.
* **Pros**: Very compact single-file backups. Fast restarts and recovery. No impact on parent process I/O (done by child).
* **Cons**: Potential data loss. If Redis crashes between snapshots, all writes since the last snapshot are lost. Forking can cause minor latency spikes if the dataset is large (e.g., several gigabytes of FAISS indexes).

### 3.2 AOF (Append Only File)

AOF logs every write operation received by the server to a disk-based log file (`appendonly.aof`).

* **How it works**: The write operations are appended to the log. An background thread fsyncs the log to disk based on the policy (usually every second).
* **Pros**: Highly durable. With `appendfsync everysec`, you lose at most 1 second of writes. The log is write-only and cannot corrupt easily.
* **Cons**: Log files are significantly larger than RDB snapshots. Restarting Redis and reconstructing the DB from AOF takes longer. High disk I/O load.

### 3.3 Hybrid (RDB + AOF)

You can enable both persistence methods simultaneously. When Redis restarts, it will load the AOF file because it is guaranteed to be the most complete.

* **How it works**: RDB snapshots are taken regularly for backup/restores, while AOF logs modifications to provide durability.
* **Modern Hybrid (since Redis 4.0)**: Redis can write an AOF file that starts with an RDB-format preamble, combining the fast loading of RDB with the step-by-step logging of AOF.

### 3.4 Persistence Selection Matrix

| Metric | RDB Only | AOF Only | RDB + AOF (Recommended) | No Persistence (Pure Cache) |
| :--- | :---: | :---: | :---: | :---: |
| **Write Performance** | High | Medium | Medium-Low | **Maximum** |
| **Recovery Speed** | Fast | Slow | Fast (with hybrid preamble) | N/A (Empty start) |
| **Data Durability** | Low | High | **High** | None |
| **Disk Space Usage** | Low | High | Medium-High | None |
| **Plagiarism Detector Context** | Backup FAISS indexes | Highly active user state | Balance of sessions and FAISS recovery | Ephemeral environments/Dev VM |

---

## 4. Recommended Production Configuration

Save the following configuration as `/etc/redis/redis.conf` in your production environments. This configuration optimizes Redis for memory safety, durability, and connections.

```ini
# ==============================================================================
# RECOMMENDED PRODUCTION REDIS CONFIGURATION FOR SEMANTIC PLAGIARISM DETECTOR
# ==============================================================================

# --- NETWORK & SECURITY ---
# Only bind to local interface or secure VPC subnets
bind 127.0.0.1 ::1
protected-mode yes
port 6379

# Close client connections after being idle for 5 minutes (0 to disable)
timeout 300

# Keep TCP connections alive, checking every 5 minutes
tcp-keepalive 300

# --- PERFORMANCE & RESOURCE LIMITS ---
# Limit the maximum number of concurrent client connections
maxclients 10000

# Explicit memory limit (e.g., 2GB). Adjust based on VM size.
maxmemory 2gb

# Evict any key using LRU when memory limit is hit (essential for caching FAISS/sessions)
maxmemory-policy allkeys-lru

# --- PERSISTENCE ---
# Enable RDB snapshotting as a fallback backup mechanism
# Save if 1 key changes in 15 mins, 10 in 5 mins, or 10000 in 1 min
save 900 1
save 300 10
save 60 10000

# Enable AOF persistence for session state and lockout state durability
appendonly yes
appendfilename "appendonly.aof"

# Fsync strategy: balance between durability and write performance
appendfsync everysec

# Avoid fsync bottlenecks during heavy background RDB snapshots
no-appendfsync-on-rewrite yes

# Auto-rewrite AOF file when size grows by 100% (minimum size 64mb)
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Directory to save database dumps and appendonly logs
dir /var/lib/redis
dbfilename dump.rdb

# --- ACTIVE DEFRAGMENTATION ---
# Dynamically reclaim memory holes caused by intensive pickle serialization / FAISS indexing
activedefrag yes
```

---

## 5. Monitoring and Optimization

To ensure Redis remains healthy, you must track memory usage, throughput, and error rates using administrative commands.

### 5.1 Crucial Diagnostic Commands

Run these commands using the `redis-cli`:

* **Check Memory Usage Details**:

  ```bash
  redis-cli INFO memory
  ```

  *Key metric to monitor:* `used_memory_human` (actual data size) and `used_memory_peak_human` (highest memory usage recorded).

* **Check Fragmentation Ratio**:

  ```bash
  redis-cli INFO memory | grep fragmentation
  ```

  *Key metric to monitor:* `mem_fragmentation_ratio`.
  * If the ratio is **> 1.5**, your system has significant memory fragmentation. Enabling `activedefrag yes` will clean this up online.
  * If the ratio is **< 1.0**, the host operating system has run out of physical memory and has started swapping to disk, causing severe latency spikes. Increase VM memory immediately.

* **Find Slow Queries**:

  ```bash
  redis-cli SLOWLOG GET 10
  ```

  Logs commands that exceeded the execution limit (typically 10 milliseconds). Helpful for detecting expensive keyspace searches or operations on huge serialized pickles.

* **Monitor Live Commands**:

  ```bash
  redis-cli MONITOR
  ```

  Outputs every command processed by the Redis server in real-time. Use sparingly in production as it increases CPU overhead.

### 5.2 Key Metrics to Watch

| Metric | Target Value | Action if Out of Range |
| :--- | :--- | :--- |
| `used_memory` | `< 90%` of `maxmemory` | Trigger alerts, clean stale sessions, or upgrade Redis VM RAM. |
| `evicted_keys` | Stable / Slow growth | A sudden spike means `maxmemory` is too small for current concurrent users. |
| `keyspace_hits` / `keyspace_misses` | High hit ratio (`hits / (hits + misses) > 85%`) | Low hit rates indicate premature eviction or misconfigured TTL policies. |
| `latency` | `< 5ms` (measured via `redis-cli --latency`) | Investigate network congestion or slow client parsing (e.g., massive JSON payloads). |

---

## 6. Best Practices

To maximize Redis efficiency in the **Semantic Plagiarism Detector** environment, implement the following operational guidelines:

1. **Avoid Staging Huge Payloads in Session Caches**:
   The FAISS indices can exceed several hundred megabytes depending on document corpus size. Ensure that raw FAISS indices are parsed and written as binary bytes directly, rather than serialized within bulky Python dictionaries. Keep session variables lightweight.

2. **Implement Connection Pooling**:
   Always reuse connections instead of opening a new socket connection for every request. The Python code uses a singleton pattern `_cache = RedisCache()` with internal socket pooling which prevents socket exhaustion.

3. **Handle Connection Failures Gracefully**:
   Redis caches are volatile by nature. As implemented in `redis_cache.py`, the system must always fall back to a thread-safe local in-memory dict when Redis is unavailable, avoiding application crashes.

4. **Disable Swap on Linux Hosts**:
   Swap memory degrades Redis throughput to disk speeds. Disable swap entirely or adjust `vm.swappiness = 1` on the Redis VM to force the OS to keep Redis pages in physical memory.

5. **Ensure Overcommit Memory is Enabled**:
   When background snapshotting is active, Linux may reject memory fork allocations. Fix this by adding `vm.overcommit_memory = 1` to `/etc/sysctl.conf`.

---

## 7. Official References

For deeper configuration details and troubleshooting, consult the official Redis documentation:

* [Redis Memory Optimization](https://redis.io/docs/latest/develop/optimization/memory-optimization/)
* [Redis Eviction Policies and Key Eviction](https://redis.io/docs/latest/develop/reference/eviction/)
* [Redis Persistence Guide](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)
* [Redis Command Reference](https://redis.io/docs/latest/commands/)
* [Redis Administration Guide](https://redis.io/docs/latest/operate/oss_and_stack/management/admin/)

### 6.1 Payload Compression Wire Format

`PayloadCompressor` stores serialized cache payloads using the following wire format:

* **Compressed payloads:** `MAGIC_HEADER + zlib_compressed_data`
* **Uncompressed payloads:** raw serialized bytes
* `MAGIC_HEADER` is `b"ZLIB_COMPRESSED_V1::"` and identifies compressed entries.
* Compression is applied when the serialized payload size is at least `COMPRESSION_THRESHOLD_BYTES`, which is **64 KiB (`64 * 1024` bytes)**.
* Consumers reading Redis entries directly should check for `MAGIC_HEADER` before attempting zlib decompression.
