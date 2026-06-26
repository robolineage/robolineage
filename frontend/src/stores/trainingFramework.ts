import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface TrainingFrameworkStatus {
  active: boolean
  current_run?: string | null
  last_error?: string | null
  root?: string
}

export interface TrainingRunItem {
  run_id: string
  run_dir: string
  status: string
  framework?: string | null
  dataset_version?: string | null
  policy_version?: string | null
  selected_rollout_count?: number | null
  selection_id?: string | null
  selection_path?: string | null
  framework_profile_path?: string | null
  checkpoint_path?: string | null
  dataset_adapt_status?: string | null
  dataset_health_status?: string | null
  dataset_health_understanding_status?: string | null
  training_monitor_understanding_status?: string | null
  target_dataset_kind?: string | null
  tmux_session?: string | null
  launch_mode?: string | null
  remote_host?: string | null
  remote_train_log?: string | null
  remote_dataset_dir?: string | null
  remote_checkpoint_dir?: string | null
  eval_success_rate?: number | null
  deploy_decision?: string | null
  updated_at?: string | null
}

export interface TrainingRunDetail {
  run: TrainingRunItem
  training_result: Record<string, unknown> | null
  training_status: Record<string, unknown> | null
  deployment_recommendation: Record<string, unknown> | null
  policy_context: Record<string, unknown> | null
  dataset_adapt_status: Record<string, unknown> | null
  dataset_adapt_result: Record<string, unknown> | null
  dataset_health_report: Record<string, unknown> | null
  dataset_health_understanding: Record<string, unknown> | null
  dataset_health_report_md: string
  training_monitor_report: Record<string, unknown> | null
  training_monitor_understanding: Record<string, unknown> | null
  training_monitor_report_md: string
  train_manifest: Record<string, unknown>[]
  dataset_log: string
  train_log: string
  eval_log: string
}

export interface FrameworkDiscoveryInput {
  target_dataset_format: string
  command_context: string
  repo_location?: string
  repo_root?: string
  remote_ssh_host?: string
  remote_repo_root?: string
  name?: string
  framework_type?: string
  dataset_command?: string
  train_command?: string
  eval_command?: string
  fixed_input_dir?: string
  checkpoint_glob?: string
  train_log?: string
  eval_result?: string
  train_launch_mode?: string
  terminal_command?: string
  terminal_hold_open?: boolean
  enable_llm_understanding?: boolean
}

export const useTrainingFrameworkStore = defineStore('trainingFramework', () => {
  const status = ref<TrainingFrameworkStatus | null>(null)
  const runs = ref<TrainingRunItem[]>([])
  const selected = ref<TrainingRunDetail | null>(null)
  const loading = ref(false)
  const detailLoading = ref(false)
  const runningDemo = ref(false)
  const discovering = ref(false)
  const discoveryResult = ref<Record<string, unknown> | null>(null)
  const discoveryEvents = ref<Record<string, unknown>[]>([])
  const error = ref<string | null>(null)

  async function refresh(limit = 30) {
    loading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/training-framework/runs?limit=${limit}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      status.value = data.status ?? null
      runs.value = data.runs ?? []
      if (selected.value) {
        const stillExists = runs.value.some((item) => item.run_id === selected.value?.run.run_id)
        if (stillExists) await selectRun(selected.value.run.run_id)
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function selectRun(runId: string) {
    detailLoading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/training-framework/runs/${encodeURIComponent(runId)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      selected.value = data as TrainingRunDetail
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      detailLoading.value = false
    }
  }

  async function runDemo() {
    runningDemo.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/training-framework/run-demo', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      await refresh()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      runningDemo.value = false
    }
  }

  async function discover(payload: FrameworkDiscoveryInput) {
    discovering.value = true
    error.value = null
    discoveryEvents.value = [
      { event: 'submitted', created_at: new Date().toISOString() },
      { event: 'waiting_for_discovery_agent', created_at: new Date().toISOString() },
    ]
    try {
      const resp = await fetch('/api/session/training-framework/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      if (data.job_id && !['completed', 'failed'].includes(String(data.status || ''))) {
        discoveryResult.value = data
        discoveryEvents.value = Array.isArray(data.events) ? data.events : discoveryEvents.value
        return await pollDiscoveryJob(String(data.job_id))
      }
      applyDiscoveryPayload(data)
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      discoveryEvents.value = [
        ...discoveryEvents.value,
        { event: 'failed', created_at: new Date().toISOString(), error: error.value },
      ]
      return null
    } finally {
      discovering.value = false
    }
  }

  async function pollDiscoveryJob(jobId: string) {
    while (true) {
      await new Promise((resolve) => window.setTimeout(resolve, 1000))
      const resp = await fetch(`/api/session/training-framework/discovery/${encodeURIComponent(jobId)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      applyDiscoveryPayload(data)
      if (data.status === 'completed') {
        const result = data.result && typeof data.result === 'object'
          ? { ...data.result, events: Array.isArray(data.events) ? data.events : data.result.events }
          : data
        discoveryResult.value = result
        return result
      }
      if (data.status === 'failed') {
        throw new Error(data.error ?? 'framework discovery failed')
      }
    }
  }

  function applyDiscoveryPayload(data: Record<string, unknown>) {
    const result = data.result && typeof data.result === 'object'
      ? data.result as Record<string, unknown>
      : data
    discoveryResult.value = {
      ...result,
      job_id: data.job_id,
      job_status: data.status,
      events: Array.isArray(data.events) ? data.events : result.events,
    }
    discoveryEvents.value = Array.isArray(data.events)
      ? data.events as Record<string, unknown>[]
      : Array.isArray(result.events)
        ? result.events as Record<string, unknown>[]
        : discoveryEvents.value
  }

  return {
    status,
    runs,
    selected,
    loading,
    detailLoading,
    runningDemo,
    discovering,
    discoveryResult,
    discoveryEvents,
    error,
    refresh,
    selectRun,
    runDemo,
    discover,
  }
})
