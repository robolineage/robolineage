import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface TaskItem {
  task_id: string
  display_name: string
  task_description?: string | null
  task_dir: string
  active: boolean
  robot?: string | null
  created_at?: string | null
  latest_task_config_version?: string | null
  latest_task_config_path?: string | null
  latest_task_config?: Record<string, unknown> | null
  rollout_count: number
  collection_session_count: number
  deployment_session_count?: number
  training_run_count: number
}

export interface CollectionSessionItem {
  session_id: string
  session_dir: string
  started_at?: string | null
  completed_at?: string | null
  rollout_count: number
  success_count: number
  dataset_decision_counts: Record<string, number>
  rollout_ids: string[]
}

export interface DeploymentSessionItem {
  session_id: string
  session_dir: string
  policy_version?: string | null
  mode?: string | null
  rollout_count: number
  success_count: number
  failure_count: number
  success_rate?: number | null
  decision?: string | null
  gating_result?: string | null
  recommended_mode?: string | null
  operator_brief?: string | null
  governance_understanding_status?: string | null
  created_at?: string | null
  rollout_ids: string[]
}

export interface TrainingSelectionItem {
  selection_id: string
  selection_path: string
  source_collection_session_ids: string[]
  include_decisions: string[]
  selected_rollout_count: number
  rollout_ids: string[]
  created_at?: string | null
  note?: string | null
}

export interface FrameworkProfileItem {
  profile_id: string
  profile_path: string
  name?: string | null
  framework_type?: string | null
  repo_root?: string | null
  created_at?: string | null
  report_path?: string | null
  understanding_status?: string | null
  understanding_report_path?: string | null
  events?: Record<string, unknown>[]
  events_path?: string | null
  target_contract?: Record<string, unknown>
  adapter_registry?: Record<string, unknown>
  dataset_adapter?: Record<string, unknown>
  integration_manifest?: Record<string, unknown>
  integration_manifest_path?: string | null
}

export interface PolicyItem {
  policy_version: string
  policy_meta_path: string
  run_id: string
  trained_on_dataset?: string | null
  architecture?: string | null
  checkpoint_path?: string | null
  eval_success_rate?: number | null
  deploy_decision?: string | null
  created_at?: string | null
}

export const useLifecycleStore = defineStore('lifecycle', () => {
  const tasks = ref<TaskItem[]>([])
  const activeTaskId = ref<string | null>(null)
  const collectionSessions = ref<CollectionSessionItem[]>([])
  const selectedCollectionDetail = ref<Record<string, unknown> | null>(null)
  const deploymentSessions = ref<DeploymentSessionItem[]>([])
  const selectedDeploymentDetail = ref<Record<string, unknown> | null>(null)
  const trainingSelections = ref<TrainingSelectionItem[]>([])
  const frameworkProfiles = ref<FrameworkProfileItem[]>([])
  const policies = ref<PolicyItem[]>([])
  const loading = ref(false)
  const detailLoading = ref(false)
  const creatingSelection = ref(false)
  const adaptingTrainingData = ref(false)
  const startingTraining = ref(false)
  const activatingTask = ref(false)
  const error = ref<string | null>(null)

  async function refreshTasks() {
    loading.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/tasks')
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      tasks.value = data.tasks ?? []
      activeTaskId.value = data.active_task_id ?? tasks.value.find((item) => item.active)?.task_id ?? null
      if (activeTaskId.value) await refreshTask(activeTaskId.value)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function activateTask(taskId: string) {
    activatingTask.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/activate`, { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      activeTaskId.value = data.task?.task_id ?? taskId
      await refreshTasks()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      activatingTask.value = false
    }
  }

  async function createTask(payload: {
    task_description?: string
    display_name?: string
    task_id?: string
  }) {
    activatingTask.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      activeTaskId.value = data.task?.task_id ?? null
      await refreshTasks()
      return data.task ?? null
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      activatingTask.value = false
    }
  }

  async function refreshTask(taskId = activeTaskId.value) {
    if (!taskId) return
    error.value = null
    try {
      const [sessionsResp, deploymentsResp, selectionsResp, profilesResp, policiesResp] = await Promise.all([
        fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/collection-sessions`),
        fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/deployment-sessions`),
        fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/training-selections`),
        fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/framework-profiles`),
        fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/policies`),
      ])
      const sessions = await sessionsResp.json()
      const deployments = await deploymentsResp.json()
      const selections = await selectionsResp.json()
      const profiles = await profilesResp.json()
      const policyPayload = await policiesResp.json()
      for (const [resp, data] of [[sessionsResp, sessions], [deploymentsResp, deployments], [selectionsResp, selections], [profilesResp, profiles], [policiesResp, policyPayload]] as const) {
        if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      }
      collectionSessions.value = sessions.sessions ?? []
      deploymentSessions.value = deployments.sessions ?? []
      trainingSelections.value = selections.selections ?? []
      frameworkProfiles.value = profiles.profiles ?? []
      policies.value = policyPayload.policies ?? []
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function selectCollectionSession(sessionId: string, taskId = activeTaskId.value) {
    if (!taskId) return
    detailLoading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/collection-sessions/${encodeURIComponent(sessionId)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      selectedCollectionDetail.value = data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      detailLoading.value = false
    }
  }

  async function selectDeploymentSession(sessionId: string, taskId = activeTaskId.value) {
    if (!taskId) return
    detailLoading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/deployment-sessions/${encodeURIComponent(sessionId)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      selectedDeploymentDetail.value = data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      detailLoading.value = false
    }
  }

  async function createSelection(payload: {
    source_collection_session_ids?: string[]
    rollout_ids?: string[]
    include_decisions?: string[]
    note?: string
  }, taskId = activeTaskId.value) {
    if (!taskId) return null
    creatingSelection.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/training-selections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      await refreshTask(taskId)
      return data.selection ?? null
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      creatingSelection.value = false
    }
  }

  async function startTraining(payload: {
    selection_id: string
    framework_profile_id: string
    policy_version: string
    architecture?: string
    run_id?: string
    train_launch_mode?: string
    remote_host?: string
    remote_repo_root?: string
    remote_dataset_dir?: string
    remote_checkpoint_dir?: string
    remote_work_dir?: string
    remote_train_log?: string
    remote_ssh_args?: string
    remote_rsync_args?: string
    remote_sync_checkpoints?: boolean
  }, taskId = activeTaskId.value) {
    if (!taskId) return null
    startingTraining.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/training-runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      startingTraining.value = false
    }
  }

  async function adaptTrainingData(payload: {
    selection_id: string
    framework_profile_id: string
    policy_version: string
    architecture?: string
  }, taskId = activeTaskId.value) {
    if (!taskId) return null
    adaptingTrainingData.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/tasks/${encodeURIComponent(taskId)}/training-data-adapt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      adaptingTrainingData.value = false
    }
  }

  return {
    tasks,
    activeTaskId,
    collectionSessions,
    selectedCollectionDetail,
    deploymentSessions,
    selectedDeploymentDetail,
    trainingSelections,
    frameworkProfiles,
    policies,
    loading,
    detailLoading,
    creatingSelection,
    adaptingTrainingData,
    startingTraining,
    activatingTask,
    error,
    refreshTasks,
    refreshTask,
    createTask,
    activateTask,
    selectCollectionSession,
    selectDeploymentSession,
    createSelection,
    adaptTrainingData,
    startTraining,
  }
})
