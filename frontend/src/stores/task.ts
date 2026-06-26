import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface TaskResult {
  task_description: string
  phases: string[]
  phase_definitions: Record<string, string>
  failure_signals: string[]
  phase_action_hints?: Record<string, unknown>
  phase_visual_hints?: Record<string, unknown>
  task_config_version?: string | null
  task_config_path?: string | null
  task_config_latest_path?: string | null
  task_config_created_at?: string | null
}

export interface RolloutState {
  active: boolean
  capture_active?: boolean
  analysis_draining?: boolean
  draining_rollouts?: Array<Record<string, unknown>>
  rollout_id: string | null
  rollout_dir: string | null
  output_jsonl: string | null
  task_config_path: string | null
  task_config_version?: string | null
  task_config_created_at?: string | null
  memory_debug_log?: string | null
  last_error: string | null
}

export interface EvalReviewState {
  active?: boolean
  queue_size?: number
  current_rollout?: string | null
  queued_rollouts?: string[]
}

export interface RolloutSessionState {
  active: boolean
  status?: 'idle' | 'active' | 'finalizing' | 'completed' | 'finalization_failed'
  finalizing?: boolean
  accepting_rollouts?: boolean
  kind: 'collection' | 'deployment' | null
  session_id: string | null
  policy_version: string | null
  started_at?: string | null
  stop_requested_at?: string | null
  finalization_stage?: string | null
  finalization_error?: string | null
  rollout_count: number
  rollout_ids: string[]
  post_review?: EvalReviewState
  eval_review?: EvalReviewState
  summary?: Record<string, unknown> | null
}

const idleRollout = (): RolloutState => ({
  active: false,
  capture_active: false,
  analysis_draining: false,
  draining_rollouts: [],
  rollout_id: null,
  rollout_dir: null,
  output_jsonl: null,
  task_config_path: null,
  task_config_version: null,
  task_config_created_at: null,
  memory_debug_log: null,
  last_error: null,
})

const idleRolloutSession = (): RolloutSessionState => ({
  active: false,
  status: 'idle',
  finalizing: false,
  accepting_rollouts: false,
  kind: null,
  session_id: null,
  policy_version: null,
  stop_requested_at: null,
  finalization_stage: null,
  finalization_error: null,
  rollout_count: 0,
  rollout_ids: [],
  post_review: { active: false, queue_size: 0 },
  eval_review: { active: false, queue_size: 0 },
  summary: null,
})

export const useTaskStore = defineStore('task', () => {
  const description = ref('')
  const result = ref<TaskResult | null>(null)
  const rollout = ref<RolloutState>(idleRollout())
  const rolloutSession = ref<RolloutSessionState>(idleRolloutSession())
  const lastSessionSummary = ref<Record<string, unknown> | null>(null)
  const loading = ref(false)
  const starting = ref(false)
  const stopping = ref(false)
  const startingCollection = ref(false)
  const stoppingCollection = ref(false)
  const startingDeployment = ref(false)
  const stoppingDeployment = ref(false)
  const error = ref<string | null>(null)

  function applyRolloutState(data: Partial<RolloutState>) {
    rollout.value = {
      ...idleRollout(),
      ...rollout.value,
      ...data,
      active: Boolean(data.active),
      capture_active: Boolean(data.capture_active ?? data.active),
      analysis_draining: Boolean(data.analysis_draining),
      draining_rollouts: Array.isArray(data.draining_rollouts) ? data.draining_rollouts : [],
      rollout_id: data.rollout_id ?? rollout.value.rollout_id ?? null,
      rollout_dir: data.rollout_dir ?? rollout.value.rollout_dir ?? null,
      output_jsonl: data.output_jsonl ?? rollout.value.output_jsonl ?? null,
      task_config_path: data.task_config_path ?? rollout.value.task_config_path ?? null,
      task_config_version: data.task_config_version ?? rollout.value.task_config_version ?? null,
      task_config_created_at: data.task_config_created_at ?? rollout.value.task_config_created_at ?? null,
      memory_debug_log: data.memory_debug_log ?? rollout.value.memory_debug_log ?? null,
      last_error: data.last_error ?? null,
    }
    const maybeSession = (data as { rollout_session?: Partial<RolloutSessionState> }).rollout_session
    if (maybeSession) applyRolloutSessionState(maybeSession)
  }

  function applyRolloutSessionState(data: Partial<RolloutSessionState>) {
    const status = data.status ?? (data.active ? 'active' : 'idle')
    const summary = data.summary ?? null
    if (summary) lastSessionSummary.value = summary
    rolloutSession.value = {
      ...idleRolloutSession(),
      ...rolloutSession.value,
      ...data,
      active: Boolean(data.active),
      status,
      finalizing: Boolean(data.finalizing ?? status === 'finalizing'),
      accepting_rollouts: Boolean(data.accepting_rollouts ?? (data.active && status === 'active')),
      kind: data.kind ?? null,
      session_id: data.session_id ?? null,
      policy_version: data.policy_version ?? null,
      stop_requested_at: data.stop_requested_at ?? null,
      finalization_stage: data.finalization_stage ?? null,
      finalization_error: data.finalization_error ?? null,
      rollout_count: Number(data.rollout_count ?? data.rollout_ids?.length ?? 0),
      rollout_ids: Array.isArray(data.rollout_ids) ? data.rollout_ids : [],
      post_review: data.post_review ?? rolloutSession.value.post_review ?? { active: false, queue_size: 0 },
      eval_review: data.eval_review ?? rolloutSession.value.eval_review ?? { active: false, queue_size: 0 },
      summary,
    }
  }

  async function refreshRollout() {
    try {
      const resp = await fetch('/api/session/task/rollout/state')
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      applyRolloutState(data)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function refreshRolloutSession() {
    try {
      const resp = await fetch('/api/session/task/session/state')
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      applyRolloutSessionState(data)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function startRollout() {
    starting.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/task/rollout/start', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      applyRolloutState({ ...data, active: data.active ?? true })
      await refreshRolloutSession()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      starting.value = false
    }
  }

  async function stopRollout() {
    stopping.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/task/rollout/stop', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) {
        throw new Error(data.error ?? `HTTP ${resp.status}`)
      }
      applyRolloutState({ ...data, active: data.active ?? false })
      await refreshRolloutSession()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      stopping.value = false
    }
  }

  async function generate() {
    if (!description.value.trim()) return
    loading.value = true
    error.value = null
    result.value = null
    try {
      const resp = await fetch('/api/session/task/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_description: description.value.trim() }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      result.value = data as TaskResult
      await refreshRollout()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function startCollectionSession() {
    startingCollection.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/task/session/collection/start', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      lastSessionSummary.value = null
      applyRolloutSessionState(data)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      startingCollection.value = false
    }
  }

  async function stopCollectionSession() {
    stoppingCollection.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/task/session/collection/stop', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      lastSessionSummary.value = (data.summary as Record<string, unknown> | undefined) ?? null
      applyRolloutSessionState(data)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      stoppingCollection.value = false
    }
  }

  async function startDeploymentSession(policyVersion?: string) {
    startingDeployment.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/task/session/deployment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy_version: policyVersion?.trim() || undefined }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      lastSessionSummary.value = null
      applyRolloutSessionState(data)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      startingDeployment.value = false
    }
  }

  async function stopDeploymentSession() {
    stoppingDeployment.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/task/session/deployment/stop', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      lastSessionSummary.value = (data.summary as Record<string, unknown> | undefined) ?? null
      applyRolloutSessionState(data)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      stoppingDeployment.value = false
    }
  }

  return {
    description,
    result,
    rollout,
    rolloutSession,
    lastSessionSummary,
    loading,
    starting,
    stopping,
    startingCollection,
    stoppingCollection,
    startingDeployment,
    stoppingDeployment,
    error,
    generate,
    refreshRollout,
    refreshRolloutSession,
    startRollout,
    stopRollout,
    startCollectionSession,
    stopCollectionSession,
    startDeploymentSession,
    stopDeploymentSession,
  }
})
