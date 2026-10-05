# Exact UNKNOWN claim disposition (customized fork)

`python -m cron.claim_disposition` is an **explicit maintenance operation**, not a
scheduler recovery sweep, retry, `mark_job_run`, or resume. It addresses F11/GLP73:
interrupted direct execution, immutable UNKNOWN ledger, paused recurring job,
retained fire claim preventing a guarded source adoption.

## Approval and exclusion contract

Obtain the required independent source review and exact-target operational approval
before using `--apply`. Source approval alone is not a live-disposition receipt.
The caller must independently reconcile the uncertain effect and bind that evidence
(e.g. archive inventory/content, partial source removal, complete remote retrieval,
and no remaining worker) in the request. Do not delete that recovery evidence.

The caller/reviewer must verify that `work_locks` names the **actual complete
payload lock set**, covering workers/orphaned children that could still perform
effects. An arbitrary new lockfile is not exclusion. For F13 this includes its
accepted common retention lock. Hold any additional maintained admission/source
reader-writer exclusion required by the operational deployment contract. A process
snapshot or a dead parent alone does not prove orphaned work quiescent. This API
cannot discover arbitrary detached consumers that do not participate in those locks.

Only local, direct, non-handoff executions are supported. The complete ledger and
job preimages must bind the original `hostname:pid:token` fire owner; the claim
heartbeat must lie inside that execution's lifetime. The exact PID must be absent
(`kill(pid, 0)` returns ESRCH). A live/recycled PID, unavailable start fingerprint,
probe error, newer/active execution, in-process run, or held work lock refuses.
There is no TTL-based override, force flag, PID killing, or effectful probe.

## Request and commands

Use the installed reviewed fork's Python with `HERMES_HOME` explicitly selecting
the approved profile. Do not change another profile or use an unreviewed checkout
against live data. The request is an operator-reviewed JSON object with:

- `home`: canonical absolute profile home.
- `job`: **complete raw** jobs.json target record, including pause, schedule,
  next-run, counters, pins and original fire claim. Names/prefixes are not accepted.
- `execution`: complete exact executions.db row, not merely the latest row.
- `reason`: nonempty reconciliation disposition rationale; not a success claim.
- `created_at`, `expires_at`: timezone-aware timestamps; at most a 15-minute window,
  captured after the terminal execution. Refresh/re-review stale evidence rather
  than reusing an expired request.
- `evidence`: nonempty array of `{ "path": "/absolute/retained-evidence", "sha256":
  "<actual full file digest>" }`; include the independently accepted reconciliation
  and quiescence evidence, not an assertion manufactured to satisfy this API.
- `work_locks`: nonempty array of `{ "path": "/actual/existing/payload.lock",
  "device": <st_dev>, "inode": <st_ino> }`. Locks are opened without creation or
  truncation, and held exclusive/nonblocking through save and readback.

```sh
python -m cron.claim_disposition --request reviewed-request.json          # preview
python -m cron.claim_disposition --request reviewed-request.json --apply  # exactly once
python -m cron.claim_disposition --inspect <32-character-execution-id>    # readback only
```

Preview performs the same lock/admission checks but never saves job/ledger records
or creates a disposition journal. The native lock helpers may create their normal
lockfiles. Linux/POSIX flock and directory fsync are required; no degraded mode.

Control files must be regular, single-link files in the selected canonical profile;
control directories (including `cron` and `claim-dispositions`) must not be symlinks.
This includes the ledger and any existing SQLite sidecars, not only `jobs.json`.
Directory/ledger/journal identities are retained across the operation; only the
native jobs save is allowed to replace `jobs.json`. A cross-profile alias is refused,
not followed or repaired.

Native `.jobs.lock` and per-job `.fire-*.lock` files are permanent namespace objects:
the maintained lock helpers open and retain them, never unlink/replace them on
release. Custody checks compare the **held descriptors** with the current paths
before intent, after intent/hash preparation, and at save/readback/acknowledgement
boundaries. Replacement before save leaves the claim intact; detected replacement
after save leaves the durable intent unacknowledged and admission held.

These checks are drift detection, **not arbitrary-writer exclusion**. The operational
caller/reviewer must establish and retain its maintained namespace/source-writer
exclusion for the whole call, including commit and acknowledgement: exclude cleanup,
restore, installer and direct writers that could rename controls, lockfiles or their
ancestors, as well as old resident writers that still allow degraded saves. Native flock
excludes participating lock users only; it cannot stop same-UID/root renames or close
a check-to-syscall race. If that external exclusion is unavailable or unproven, do
not apply. This source patch neither creates a new maintenance gate nor proves that
runtime prerequisite on the receiving host.

Lock order: native per-job fire fence → strict native jobs registry lock → ledger
`BEGIN IMMEDIATE` → actual work locks. Ordinary `_jobs_lock` sections retain their
bounded read fallback, but the native registry saver requires an acquired cross-process
lock before staging or replacing bytes, including inside nested degraded sections.
Timeout or unavailable locking raises instead of reporting a successful write. Uncontended
and nested writes under an acquired lock retain the normal save/merge behavior.

This excludes cooperating writers running the repaired saver on the same lock inode.
It does not upgrade already-loaded code or exclude arbitrary namespace writers. Before
live use, the authorized owner must drain/retire relevant old registry writers and their
launch paths, adopt reviewed bytes, and establish fresh resident import provenance while
retaining the namespace/source and payload-worker exclusion above through acknowledgement.
Replacing files on disk, global dispatch pause, or a short operation is not that boundary.
Independent receiving acceptance and rollout remain separate from this source correction.

No ledger schema/data update or prune occurs during disposition, and no scheduler execution
is started. The native `create_execution` admission path takes the same SQLite writer
boundary before checking for an unacknowledged intent; direct `claim_job_for_fire`
(including force) also rejects it. This closes the actual provider pre-fire-attempt
race without changing provider dispatch. Ordinary unrelated jobs are not held.
Terminal pruning excludes exact execution IDs with disposition journals, including
verified journals: the original UNKNOWN survives the ordinary history limit.

Apply fsyncs a create-exclusive per-execution intent under
`<home>/cron/claim-dispositions/`, rechecks preimages/evidence/lock identities,
rechecks the clock after all hashing and lock/path preparation immediately before
native save, sends a **copy** to the native saver and compares the entire job list
plus ledger rows afterward. Expiry crossed during hashing leaves the old claim
intact and the durable intent latched; it is not permission to refresh and retry.
The sole job-record delta is `fire_claim: null`; UNKNOWN, pause,
schedules, cursors, counters, pins and unrelated records stay unchanged.

## Uncertain outcomes and repeats

Any existing intent refuses further apply, even if evidence/request bytes change.
Never delete/replace the journal to re-arm the operation. No automated retry,
rollback, success synthesis, resume, or new fire is provided. After an exception or
lost acknowledgement use `--inspect` only: `applied`, `not_applied`, or `uncertain`
describes the current full-registry/exact-ledger readback, **not** permission to
retry or a reconstructed execution outcome. Even `not_applied` keeps the intent
latched; escalate for separately reviewed recovery. A damaged intent also fails
closed. A later retention deployment remains a distinct approved operation.
