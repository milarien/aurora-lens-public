# Session durability and production Redis

Production uses the existing RedisSessionStore architecture. Licensed/local
installations retain their explicit memory default; memory is not a Redis
failure fallback.

## State and expiry

Conversational SessionRecord keys have a TTL. Governance-hold keys have no TTL.
Schema 2 carries unresolved referents, pending clarification, epistemic holds,
typed hiring criteria (including literal years and clarification evidence), and
explicit closure dispositions. Conversation expiry is not a resolution. Session
writes preserve a hold before saving the expiring record; explicit removal is
persisted before its hold key is deleted.

## Configuration and failures

Production sets AURORA_LENS_SESSION_BACKEND=redis and AURORA_LENS_REDIS_URL to a
server-only private-network connection. Credentials must never reach browsers.
Unknown backend values and unreadable Redis session records fail closed.
Missing Redis URL prevents startup; Redis connection/store failures raise
SessionStoreError rather than constructing empty memory state.

Redis uses one persistent /data volume, appendonly=yes, appendfsync=always,
aof-load-truncated=no, and maxmemory-policy=noeviction. The hold keys must not
be silently evicted. The initial deployment bounds Redis data to 64 MiB, service
memory to 0.2 GB, and service CPU to 0.03 vCPU. Resource estimates are not a
promise of indefinite capacity: capacity/persistence failures must be repaired
without discarding outstanding governance.

## Migration and evidence

The October 2026 demo migration had explicit human authorisation to discard
all existing demo and diagnostic session state. It did not infer resolution,
authority, or approval. Audit files and test evidence were retained, with a
checksum-verified pre-migration archive and a separate migration disposition.
No audit reconstruction or live-memory recovery was used.

Verify newly created holds through actual application restart, deployment,
conversation expiry and datastore restart; compare typed context and closure
journals as well as decisions. Test normal and streaming store-unavailable
requests. A simulated clock or a repository comparison does not substitute for
these production observations. Complete shared-program comparison remains a
required release gate; installation differences remain explicitly pinned.

One Redis service is durable storage, not high availability. This configuration
has no redundant replica or new paid backup plan. Persistent-volume loss is a
separate recovery incident; an empty replacement database must not be presented
as preserved session state.
