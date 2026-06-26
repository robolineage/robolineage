<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import {
  NAlert,
  NButton,
  NCard,
  NDataTable,
  NForm,
  NFormItem,
  NGrid,
  NGridItem,
  NInput,
  NSelect,
  NTabPane,
  NTabs,
  NTag,
  type DataTableColumns,
} from 'naive-ui'
import {
  useLifecycleStore,
  type CollectionSessionItem,
  type DeploymentSessionItem,
  type FrameworkProfileItem,
  type PolicyItem,
  type TaskItem,
  type TrainingSelectionItem,
} from '@/stores/lifecycle'
import { usePostReviewStore, type ReviewRollout } from '@/stores/postReview'
import { useTrainingFrameworkStore, type TrainingRunItem } from '@/stores/trainingFramework'

const lifecycle = useLifecycleStore()
const postReview = usePostReviewStore()
const training = useTrainingFrameworkStore()
const route = useRoute()

const selectedSessionIds = ref<string[]>([])
const selectedEvalRolloutId = ref('')
const selectionNote = ref('')
const activeTab = ref(typeof route.query.tab === 'string' ? route.query.tab : 'review')
let trainingRefreshTimer: number | null = null
const trainingRequest = reactive({
  selection_id: '',
  framework_profile_id: '',
  dataset_run_id: '',
  policy_version: '',
  architecture: '',
  train_launch_mode: 'remote_tmux',
  remote_host: '',
  remote_repo_root: '',
  remote_dataset_dir: '',
  remote_checkpoint_dir: '',
  remote_work_dir: '',
  remote_train_log: '',
  remote_ssh_args: '',
  remote_rsync_args: '-az --delete',
  remote_sync_checkpoints: true,
})
const discoveryForm = reactive({
  repo_location: 'local',
  repo_root: '',
  remote_ssh_host: '',
  remote_repo_root: '',
  target_dataset_format: '',
  command_context: '',
})

const activeTask = computed(() => lifecycle.tasks.find((item) => item.task_id === lifecycle.activeTaskId) ?? null)
const selectionOptions = computed(() => lifecycle.trainingSelections.map((item) => ({
  label: `${item.selection_id} (${item.selected_rollout_count})`,
  value: item.selection_id,
})))
const frameworkOptions = computed(() => lifecycle.frameworkProfiles.map((item) => ({
  label: item.name ? `${item.name} / ${item.profile_id}` : item.profile_id,
  value: item.profile_id,
})))
const adaptedDatasetOptions = computed(() => training.runs
  .filter((item) => ['completed', 'skipped'].includes(String(item.dataset_adapt_status || '')) && item.status !== 'blocked')
  .map((item) => ({
    label: `${item.run_id} / ${displayValue(item.dataset_version)} / ${displayValue(item.target_dataset_kind)}`,
    value: item.run_id,
  })))
const launchModeOptions = [
  { label: 'Remote tmux', value: 'remote_tmux' },
  { label: 'Local tmux', value: 'tmux' },
  { label: 'Inline', value: 'inline' },
]
const repoLocationOptions = [
  { label: 'Local repo', value: 'local' },
  { label: 'Remote repo', value: 'remote' },
]
const discoveryEvents = computed(() => {
  const resultEvents = training.discoveryResult?.events
  if (Array.isArray(resultEvents)) return resultEvents as Record<string, unknown>[]
  return training.discoveryEvents
})
const discoveryTargetContract = computed(() => asRecord(training.discoveryResult?.target_contract))
const discoveryDatasetAdapter = computed(() => asRecord(training.discoveryResult?.dataset_adapter))
const discoveryManifest = computed(() => asRecord(training.discoveryResult?.integration_manifest))
const discoveryManifestOutputs = computed(() => asRecord(discoveryManifest.value.outputs))
const discoveryRepoReady = computed(() => discoveryForm.repo_location === 'remote'
  ? Boolean(discoveryForm.remote_ssh_host.trim() && discoveryForm.remote_repo_root.trim())
  : Boolean(discoveryForm.repo_root.trim()))
const latestTrainingRun = computed(() => training.runs[0] ?? null)
const selectedDatasetRun = computed(() => training.runs.find((item) => item.run_id === trainingRequest.dataset_run_id) ?? null)
const monitorRun = computed(() => training.selected?.run ?? selectedDatasetRun.value ?? latestTrainingRun.value)
const startTrainingDisabled = computed(() => {
  const hasSelectedDataset = Boolean(trainingRequest.dataset_run_id && selectedDatasetRun.value)
  const hasPolicy = Boolean(String(trainingRequest.policy_version || selectedDatasetRun.value?.policy_version || '').trim())
  const hasRemote = trainingRequest.train_launch_mode !== 'remote_tmux' || Boolean(trainingRequest.remote_host.trim())
  return !hasSelectedDataset || !hasPolicy || !hasRemote
})
const selectedAdaptStatus = computed(() => asRecord(training.selected?.dataset_adapt_status))
const selectedAdaptWarnings = computed(() => {
  const warnings = selectedAdaptStatus.value.warnings
  return Array.isArray(warnings) ? warnings.map((item) => String(item)).filter(Boolean) : []
})
const selectedAdaptDisplayStatus = computed(() => {
  const runStatus = monitorRun.value?.dataset_adapt_status
  const detailStatus = selectedAdaptStatus.value.status
  if (detailStatus === 'skipped' && runStatus === 'blocked') return 'blocked'
  return detailStatus || runStatus
})
const showSelectedAdaptDetail = computed(() =>
  Boolean(selectedAdaptStatus.value.reason || selectedAdaptStatus.value.error || selectedAdaptWarnings.value.length),
)
const selectedDatasetHealthReport = computed(() => asRecord(training.selected?.dataset_health_report))
const selectedDatasetHealthUnderstanding = computed(() => asRecord(training.selected?.dataset_health_understanding))
const selectedDatasetHealthCoverage = computed(() => asRecord(selectedDatasetHealthReport.value.phase_coverage))
const selectedDatasetHealthMissingPhases = computed(() => {
  const phases = selectedDatasetHealthCoverage.value.missing_phases
  return Array.isArray(phases) ? phases.map((item) => String(item)).filter(Boolean) : []
})
const selectedTrainingMonitorUnderstanding = computed(() => asRecord(training.selected?.training_monitor_understanding))
const selectedTrainingLikelyCauses = computed(() => {
  const causes = selectedTrainingMonitorUnderstanding.value.likely_causes
  return Array.isArray(causes) ? causes.map((item) => String(item)).filter(Boolean) : []
})
const hasSelectedDatasetHealth = computed(() =>
  Object.keys(selectedDatasetHealthReport.value).length > 0 || Object.keys(selectedDatasetHealthUnderstanding.value).length > 0,
)
const hasSelectedTrainingUnderstanding = computed(() => Object.keys(selectedTrainingMonitorUnderstanding.value).length > 0)
const selectedDeploymentRollouts = computed<Record<string, unknown>[]>(() => {
  const detail = lifecycle.selectedDeploymentDetail
  const rows = detail?.rollouts
  return Array.isArray(rows) ? rows as Record<string, unknown>[] : []
})
const selectedEvalRollout = computed(() => selectedDeploymentRollouts.value.find((item) => item.rollout_id === selectedEvalRolloutId.value) ?? null)
const selectedDeploymentSession = computed(() => asRecord(lifecycle.selectedDeploymentDetail?.session))
const selectedDeploymentSummary = computed(() => asRecord(lifecycle.selectedDeploymentDetail?.summary))
const selectedDeploymentDecision = computed(() => asRecord(lifecycle.selectedDeploymentDetail?.deployment_decision))
const selectedCollectionRecommendation = computed(() => asRecord(lifecycle.selectedDeploymentDetail?.collection_recommendation))
const selectedNextCollectionBrief = computed(() => asRecord(lifecycle.selectedDeploymentDetail?.next_collection_brief))
const selectedGovernanceUnderstanding = computed(() => asRecord(lifecycle.selectedDeploymentDetail?.deployment_governance_understanding))
const selectedGovernanceRiskNotes = computed(() => {
  const notes = selectedGovernanceUnderstanding.value.risk_notes
  return Array.isArray(notes) ? notes.map((item) => String(item)).filter(Boolean) : []
})
const hasSelectedGovernanceUnderstanding = computed(() => Object.keys(selectedGovernanceUnderstanding.value).length > 0)
const selectedFocusPhases = computed(() => {
  const value = selectedCollectionRecommendation.value.focus_phases
  return Array.isArray(value) ? value.map((item) => String(item)) : []
})

watch(() => route.query.tab, (tab) => {
  if (typeof tab === 'string') activeTab.value = tab
})

watch(() => trainingRequest.dataset_run_id, (runId) => {
  const run = training.runs.find((item) => item.run_id === runId)
  if (!run) return
  if (run.selection_id) trainingRequest.selection_id = String(run.selection_id)
  if (run.framework_profile_path) {
    const profile = lifecycle.frameworkProfiles.find((item) => item.profile_path === run.framework_profile_path)
    if (profile?.profile_id) trainingRequest.framework_profile_id = profile.profile_id
  }
  if (run.policy_version) trainingRequest.policy_version = String(run.policy_version)
  void training.selectRun(run.run_id)
})

const taskColumns: DataTableColumns<TaskItem> = [
  {
    title: 'Task',
    key: 'display_name',
    render(row) {
      return h('div', { class: 'space-y-1' }, [
        h('div', { class: 'font-medium' }, row.display_name),
        h('div', { class: 'text-xs text-gray-400' }, row.task_id),
      ])
    },
  },
  { title: 'Rollouts', key: 'rollout_count', width: 100 },
  { title: 'Sessions', key: 'collection_session_count', width: 100 },
  { title: 'Training', key: 'training_run_count', width: 100 },
  {
    title: 'State',
    key: 'active',
    width: 110,
    render(row) {
      return row.active
        ? h(NTag, { type: 'success', size: 'small' }, () => 'Active')
        : h(NButton, { size: 'tiny', loading: lifecycle.activatingTask, onClick: () => lifecycle.activateTask(row.task_id) }, () => 'Activate')
    },
  },
]

const sessionColumns: DataTableColumns<CollectionSessionItem> = [
  {
    title: 'Use',
    key: 'select',
    width: 80,
    render(row) {
      const selected = selectedSessionIds.value.includes(row.session_id)
      return h(NButton, { size: 'tiny', type: selected ? 'primary' : 'default', onClick: () => toggleSession(row.session_id) }, () => selected ? 'Selected' : 'Select')
    },
  },
  { title: 'Session', key: 'session_id' },
  { title: 'Rollouts', key: 'rollout_count', width: 100 },
  { title: 'Success', key: 'success_count', width: 100 },
  {
    title: 'Updated',
    key: 'completed_at',
    width: 190,
    render(row) {
      return row.completed_at ?? row.started_at ?? '-'
    },
  },
]

const deploymentSessionColumns: DataTableColumns<DeploymentSessionItem> = [
  { title: 'Session', key: 'session_id' },
  { title: 'Policy', key: 'policy_version', width: 160 },
  { title: 'Rollouts', key: 'rollout_count', width: 95 },
  { title: 'Success', key: 'success_rate', width: 100 },
  {
    title: 'Decision',
    key: 'decision',
    width: 150,
    render(row) {
      return h(NTag, { size: 'small', type: decisionTagType(row.decision) }, () => row.decision ?? 'pending')
    },
  },
  {
    title: 'Understanding',
    key: 'governance_understanding_status',
    width: 145,
    render(row) {
      return h(NTag, { size: 'small', type: row.governance_understanding_status === 'generated' ? 'success' : 'default' }, () => row.governance_understanding_status ?? 'basic')
    },
  },
  {
    title: 'Updated',
    key: 'created_at',
    width: 190,
    render(row) {
      return row.created_at ?? '-'
    },
  },
]

const reviewColumns: DataTableColumns<ReviewRollout> = [
  { title: 'Rollout', key: 'rollout_id' },
  {
    title: 'Status',
    key: 'status',
    width: 140,
    render(row) {
      return h(NTag, { size: 'small', type: row.status === 'reviewed' ? 'success' : 'warning' }, () => row.status)
    },
  },
  {
    title: 'Dataset',
    key: 'dataset_decision',
    width: 170,
    render(row) {
      return h(NTag, { size: 'small', type: reviewDatasetTagType(row), class: 'font-medium' }, () => reviewDatasetLabel(row))
    },
  },
  {
    title: 'Train',
    key: 'accepted_for_training',
    width: 120,
    render(row) {
      return h(NTag, { size: 'small', type: trainingAdmissionTagType(row) }, () => trainingAdmissionLabel(row))
    },
  },
  { title: 'Final phase', key: 'final_phase', width: 150 },
]

const evalRolloutColumns: DataTableColumns<Record<string, unknown>> = [
  { title: 'Rollout', key: 'rollout_id' },
  {
    title: 'Success',
    key: 'final_success',
    width: 110,
    render(row) {
      return h(NTag, { size: 'small', type: row.final_success === true ? 'success' : row.final_success === false ? 'error' : 'default' }, () => displayValue(row.final_success))
    },
  },
  { title: 'Final phase', key: 'final_phase', width: 150 },
  { title: 'Next action', key: 'recommended_next_action', width: 210 },
]

const selectionColumns: DataTableColumns<TrainingSelectionItem> = [
  { title: 'Selection', key: 'selection_id' },
  { title: 'Rollouts', key: 'selected_rollout_count', width: 110 },
  { title: 'Source sessions', key: 'source_collection_session_ids' },
  { title: 'Created', key: 'created_at', width: 190 },
]

const frameworkColumns: DataTableColumns<FrameworkProfileItem> = [
  { title: 'Profile', key: 'profile_id' },
  { title: 'Name', key: 'name' },
  {
    title: 'Adapter',
    key: 'dataset_adapter',
    width: 190,
    render(row) {
      const adapter = asRecord(row.dataset_adapter)
      return h('span', {}, displayValue(adapter.adapter_id || adapter.strategy))
    },
  },
  { title: 'Repo', key: 'repo_root' },
  {
    title: 'LLM',
    key: 'understanding_status',
    width: 120,
    render(row) {
      return h(NTag, { size: 'small', type: row.understanding_status === 'generated' ? 'success' : 'default' }, () => row.understanding_status ?? 'basic')
    },
  },
]

const runColumns: DataTableColumns<TrainingRunItem> = [
  { title: 'Run', key: 'run_id' },
  { title: 'Policy', key: 'policy_version', width: 160 },
  {
    title: 'Status',
    key: 'status',
    width: 120,
    render(row) {
      return h(NTag, { size: 'small', type: row.status === 'completed' ? 'success' : 'warning' }, () => row.status)
    },
  },
  { title: 'Adapt', key: 'dataset_adapt_status', width: 110 },
  { title: 'tmux', key: 'tmux_session', width: 150 },
  { title: 'Launch', key: 'launch_mode', width: 120 },
  { title: 'Remote', key: 'remote_host', width: 170 },
  { title: 'Success', key: 'eval_success_rate', width: 100 },
  { title: 'Deploy', key: 'deploy_decision', width: 120 },
]

const policyColumns: DataTableColumns<PolicyItem> = [
  { title: 'Policy', key: 'policy_version' },
  { title: 'Architecture', key: 'architecture', width: 150 },
  { title: 'Success', key: 'eval_success_rate', width: 100 },
  { title: 'Deploy', key: 'deploy_decision', width: 120 },
]

function toggleSession(sessionId: string) {
  selectedSessionIds.value = selectedSessionIds.value.includes(sessionId)
    ? selectedSessionIds.value.filter((item) => item !== sessionId)
    : [...selectedSessionIds.value, sessionId]
}

function displayValue(value: unknown) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

function compactTraceValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => compactTraceValue(item)).join(', ')
  }
  if (value && typeof value === 'object') {
    return JSON.stringify(value)
  }
  return displayValue(value)
}

function discoveryEventDetail(event: Record<string, unknown>): string {
  const hidden = new Set(['event', 'created_at'])
  const detail = Object.entries(event)
    .filter(([key, value]) => !hidden.has(key) && value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${key}: ${compactTraceValue(value)}`)
    .join(' | ')
  return detail.length > 520 ? `${detail.slice(0, 520)}...` : detail
}

function decisionTagType(value: unknown): 'success' | 'warning' | 'error' | 'info' | 'default' {
  if (value === 'deploy_recommended') return 'success'
  if (value === 'rollback_recommended') return 'error'
  if (value === 'collect_more_data' || value === 'hold') return 'warning'
  return 'default'
}

function isFailurePoolRollout(row: ReviewRollout): boolean {
  return row.admission_class === 'failure_pool_candidate'
}

function isTrainableReviewRollout(row: ReviewRollout): boolean {
  return row.dataset_decision === 'needs_review' && row.accepted_for_training === true
}

function reviewDatasetTagType(row: ReviewRollout): 'success' | 'warning' | 'error' | 'info' | 'default' {
  if (isFailurePoolRollout(row)) return 'error'
  if (isTrainableReviewRollout(row)) return 'warning'
  if (row.dataset_decision === 'accepted') return 'success'
  if (row.dataset_decision === 'retry_recommended' || row.dataset_decision === 'needs_review') return 'warning'
  if (row.dataset_decision === 'rejected') return 'error'
  return 'default'
}

function reviewDatasetLabel(row: ReviewRollout): string {
  if (isFailurePoolRollout(row)) return 'failure pool'
  if (isTrainableReviewRollout(row)) return 'review + trainable'
  return row.dataset_decision ?? 'pending'
}

function trainingAdmissionTagType(row: ReviewRollout): 'success' | 'warning' | 'error' | 'default' {
  if (row.accepted_for_training === true) return row.requires_review ? 'warning' : 'success'
  if (row.accepted_for_training === false) return 'error'
  return 'default'
}

function trainingAdmissionLabel(row: ReviewRollout): string {
  if (row.accepted_for_training === true) return row.requires_review ? 'trainable review' : 'trainable'
  if (row.accepted_for_training === false) return isFailurePoolRollout(row) ? 'failure hold' : 'hold'
  return 'pending'
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

async function refreshAll() {
  await Promise.all([lifecycle.refreshTasks(), postReview.refresh(), training.refresh()])
  const sessionId = typeof route.query.session === 'string' ? route.query.session : ''
  if (activeTab.value === 'evaluation' && sessionId) {
    await selectDeploymentSession(sessionId)
  }
}

async function createTrainingSelection() {
  const selection = await lifecycle.createSelection({
    source_collection_session_ids: selectedSessionIds.value,
    include_decisions: ['accepted', 'needs_review'],
    note: selectionNote.value,
  })
  if (selection?.selection_id) trainingRequest.selection_id = selection.selection_id
}

async function discoverFramework() {
  const result = await training.discover({
    ...discoveryForm,
    repo_root: discoveryForm.repo_location === 'local' ? discoveryForm.repo_root : undefined,
    remote_ssh_host: discoveryForm.repo_location === 'remote' ? discoveryForm.remote_ssh_host : undefined,
    remote_repo_root: discoveryForm.repo_location === 'remote' ? discoveryForm.remote_repo_root : undefined,
  })
  if (result && lifecycle.activeTaskId) await lifecycle.refreshTask(lifecycle.activeTaskId)
}

async function adaptTrainingData() {
  const result = await lifecycle.adaptTrainingData({
    selection_id: trainingRequest.selection_id,
    framework_profile_id: trainingRequest.framework_profile_id,
    policy_version: trainingRequest.policy_version,
    architecture: trainingRequest.architecture || undefined,
  })
  if (result?.run_id) {
    trainingRequest.dataset_run_id = String(result.run_id)
    await training.refresh()
    await training.selectRun(String(result.run_id))
  }
}

async function startTraining() {
  const selectedRun = selectedDatasetRun.value
  const result = await lifecycle.startTraining({
    selection_id: trainingRequest.selection_id || String(selectedRun?.selection_id || ''),
    framework_profile_id: trainingRequest.framework_profile_id,
    policy_version: trainingRequest.policy_version || String(selectedRun?.policy_version || ''),
    architecture: trainingRequest.architecture || undefined,
    run_id: trainingRequest.dataset_run_id || undefined,
    train_launch_mode: trainingRequest.train_launch_mode || undefined,
    remote_host: trainingRequest.remote_host || undefined,
    remote_repo_root: trainingRequest.remote_repo_root || undefined,
    remote_dataset_dir: trainingRequest.remote_dataset_dir || undefined,
    remote_checkpoint_dir: trainingRequest.remote_checkpoint_dir || undefined,
    remote_work_dir: trainingRequest.remote_work_dir || undefined,
    remote_train_log: trainingRequest.remote_train_log || undefined,
    remote_ssh_args: trainingRequest.remote_ssh_args || undefined,
    remote_rsync_args: trainingRequest.remote_rsync_args || undefined,
    remote_sync_checkpoints: trainingRequest.remote_sync_checkpoints,
  })
  if (result?.run_id) {
    await training.refresh()
    await training.selectRun(String(result.run_id))
  }
  await refreshAll()
}

async function selectDeploymentSession(sessionId: string) {
  await lifecycle.selectDeploymentSession(sessionId)
  const first = selectedDeploymentRollouts.value[0]
  selectedEvalRolloutId.value = first?.rollout_id ? String(first.rollout_id) : ''
}

function selectEvalRollout(row: Record<string, unknown>) {
  selectedEvalRolloutId.value = row.rollout_id ? String(row.rollout_id) : ''
}

onMounted(() => {
  void refreshAll()
  trainingRefreshTimer = window.setInterval(() => {
    if (activeTab.value !== 'training' && !training.status?.active) return
    void training.refresh()
  }, 3000)
})

onBeforeUnmount(() => {
  if (trainingRefreshTimer !== null) {
    window.clearInterval(trainingRefreshTimer)
    trainingRefreshTimer = null
  }
})
</script>

<template>
  <div class="space-y-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 class="text-2xl font-semibold">Review / Lifecycle</h1>
        <p class="mt-1 text-sm text-gray-500">Inspect tasks, reviews, training selections, frameworks, and policy versions.</p>
      </div>
      <NButton :loading="lifecycle.loading || postReview.loading || training.loading" @click="refreshAll()">Refresh</NButton>
    </div>

    <NAlert v-if="lifecycle.error || postReview.error || training.error" type="error" closable>
      {{ lifecycle.error || postReview.error || training.error }}
    </NAlert>

    <NGrid cols="1 xl:3" :x-gap="16" :y-gap="16" responsive="screen">
      <NGridItem>
        <NCard title="Tasks" :bordered="false">
          <NDataTable :columns="taskColumns" :data="lifecycle.tasks" :loading="lifecycle.loading" :pagination="{ pageSize: 6 }" />
        </NCard>
      </NGridItem>
      <NGridItem span="1 xl:2">
        <NCard title="Active Task" :bordered="false">
          <div v-if="activeTask" class="grid gap-2 text-sm md:grid-cols-2">
            <div><span class="text-gray-500">Task</span><div class="font-medium">{{ activeTask.display_name }}</div></div>
            <div><span class="text-gray-500">Robot</span><div class="font-medium">{{ activeTask.robot ?? '-' }}</div></div>
            <div><span class="text-gray-500">Rollouts</span><div class="font-medium">{{ activeTask.rollout_count }}</div></div>
            <div><span class="text-gray-500">Training runs</span><div class="font-medium">{{ activeTask.training_run_count }}</div></div>
          </div>
          <div v-else class="text-sm text-gray-500">No active task.</div>
        </NCard>
      </NGridItem>
    </NGrid>

    <NTabs v-model:value="activeTab" type="line" animated>
      <NTabPane name="review" tab="Post Review">
        <NGrid cols="1 xl:2" :x-gap="16" :y-gap="16" responsive="screen">
          <NGridItem>
            <NCard title="Reviewed Rollouts" :bordered="false">
              <NDataTable
                :columns="reviewColumns"
                :data="postReview.rollouts"
                :loading="postReview.loading"
                :pagination="{ pageSize: 8 }"
                :row-props="(row) => ({ class: 'cursor-pointer', onClick: () => postReview.selectRollout(row.rollout_id) })"
              />
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard title="Selected Review" :bordered="false">
              <div v-if="postReview.selected" class="space-y-3">
                <div class="flex flex-wrap gap-2">
                  <NTag type="info">{{ postReview.selected.rollout.rollout_id }}</NTag>
                  <NTag :type="reviewDatasetTagType(postReview.selected.rollout)" class="font-medium">
                    {{ reviewDatasetLabel(postReview.selected.rollout) }}
                  </NTag>
                  <NTag :type="trainingAdmissionTagType(postReview.selected.rollout)">
                    {{ trainingAdmissionLabel(postReview.selected.rollout) }}
                  </NTag>
                  <NTag
                    v-if="postReview.selected.rollout.admission_class && !isFailurePoolRollout(postReview.selected.rollout)"
                    type="default"
                  >
                    {{ postReview.selected.rollout.admission_class }}
                  </NTag>
                </div>
                <pre class="max-h-80 overflow-auto rounded bg-gray-50 p-3 text-xs whitespace-pre-wrap">{{ postReview.selected.review_report || 'No report yet.' }}</pre>
              </div>
              <div v-else class="text-sm text-gray-500">Select a rollout on the left to inspect annotation, failure analysis, and dataset admission.</div>
            </NCard>
          </NGridItem>
        </NGrid>
      </NTabPane>

      <NTabPane name="selection" tab="Dataset Selection">
        <NGrid cols="1 xl:2" :x-gap="16" :y-gap="16" responsive="screen">
          <NGridItem>
            <NCard title="Collection Sessions" :bordered="false">
              <NDataTable :columns="sessionColumns" :data="lifecycle.collectionSessions" :pagination="{ pageSize: 8 }" />
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard title="Create Training Selection" :bordered="false">
              <NForm label-placement="top">
                <NFormItem label="Selected sessions">
                  <NInput :value="selectedSessionIds.join(', ')" readonly placeholder="Select collection sessions on the left" />
                </NFormItem>
                <NFormItem label="Note">
                  <NInput v-model:value="selectionNote" type="textarea" placeholder="Optional note for this data batch" />
                </NFormItem>
                <NButton
                  type="primary"
                  :disabled="selectedSessionIds.length === 0"
                  :loading="lifecycle.creatingSelection"
                  @click="createTrainingSelection"
                >
                  Create Training Selection
                </NButton>
              </NForm>
              <NDataTable class="mt-4" :columns="selectionColumns" :data="lifecycle.trainingSelections" :pagination="{ pageSize: 6 }" />
            </NCard>
          </NGridItem>
        </NGrid>
      </NTabPane>

      <NTabPane name="training" tab="Training">
        <NGrid cols="1 xl:2" :x-gap="16" :y-gap="16" responsive="screen">
          <NGridItem>
            <NCard title="Framework Discovery" :bordered="false">
              <NForm label-placement="top">
                <NFormItem label="Repository location">
                  <NSelect v-model:value="discoveryForm.repo_location" :options="repoLocationOptions" />
                </NFormItem>
                <NFormItem v-if="discoveryForm.repo_location === 'local'" label="Local repo root">
                  <NInput v-model:value="discoveryForm.repo_root" placeholder="/home/robot/training_repo" />
                </NFormItem>
                <div v-else class="grid gap-2 md:grid-cols-2">
                  <NFormItem label="SSH host">
                    <NInput v-model:value="discoveryForm.remote_ssh_host" placeholder="user@training-server" />
                  </NFormItem>
                  <NFormItem label="Remote repo root">
                    <NInput v-model:value="discoveryForm.remote_repo_root" placeholder="/home/user/Transengram_datacollection-main" />
                  </NFormItem>
                </div>
                <NFormItem label="Training dataset format">
                  <NInput
                    v-model:value="discoveryForm.target_dataset_format"
                    type="textarea"
                    :autosize="{ minRows: 10, maxRows: 18 }"
                    placeholder="Format: HDF5 / RLDS / LeRobot / custom&#10;Episode layout, keys, dtypes, camera names, action/state names..."
                  />
                </NFormItem>
                <NFormItem label="Command context">
                  <NInput
                    v-model:value="discoveryForm.command_context"
                    type="textarea"
                    :autosize="{ minRows: 10, maxRows: 18 }"
                    placeholder="conda env: policy_env&#10;train command: tools/02_train.sh&#10;Any flags/config notes the training repo expects..."
                  />
                </NFormItem>
                <NButton
                  type="primary"
                  :loading="training.discovering"
                  :disabled="!discoveryRepoReady || !discoveryForm.target_dataset_format.trim() || !discoveryForm.command_context.trim()"
                  @click="discoverFramework"
                >
                  Start Discovery
                </NButton>
              </NForm>
              <div class="mt-4 rounded border border-gray-100 bg-gray-50 p-3">
                <div class="mb-2 flex items-center justify-between">
                  <span class="text-xs font-medium text-gray-500">Discovery Trace</span>
                  <NTag size="small" :type="training.discovering ? 'warning' : training.discoveryResult ? 'success' : 'default'">
                    {{ training.discovering ? 'running' : training.discoveryResult ? 'completed' : 'idle' }}
                  </NTag>
                </div>
                <div class="max-h-48 space-y-2 overflow-auto text-xs">
                  <div
                    v-for="(event, index) in discoveryEvents"
                    :key="`${displayValue(event.event)}-${index}`"
                    class="rounded bg-white px-2 py-1"
                  >
                    <div class="font-medium text-gray-700">{{ displayValue(event.event) }}</div>
                    <div class="text-gray-400">{{ displayValue(event.created_at) }}</div>
                    <div v-if="discoveryEventDetail(event)" class="break-all text-gray-600">
                      {{ discoveryEventDetail(event) }}
                    </div>
                  </div>
                  <div v-if="!discoveryEvents.length" class="text-gray-400">No discovery events yet.</div>
                </div>
                <div v-if="training.discoveryResult" class="mt-3 grid gap-2 text-xs md:grid-cols-2">
                  <div>
                    <span class="text-gray-400">Dataset kind</span>
                    <div class="font-medium text-gray-700">{{ displayValue(discoveryTargetContract.dataset_kind) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Adapter</span>
                    <div class="font-medium text-gray-700">{{ displayValue(discoveryDatasetAdapter.adapter_id || discoveryDatasetAdapter.strategy) }}</div>
                  </div>
                  <div class="md:col-span-2">
                    <span class="text-gray-400">Integration manifest</span>
                    <div class="break-all text-gray-700">{{ displayValue(training.discoveryResult.integration_manifest_path || training.discoveryResult.profile_path) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Metrics source</span>
                    <div class="font-medium text-gray-700">{{ displayValue(discoveryManifestOutputs.primary_metrics_source) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Checkpoint glob</span>
                    <div class="break-all font-medium text-gray-700">{{ displayValue(discoveryManifestOutputs.checkpoint_glob) }}</div>
                  </div>
                  <div class="md:col-span-2">
                    <span class="text-gray-400">TensorBoard logdir candidates</span>
                    <div class="break-all text-gray-700">{{ displayValue(Array.isArray(discoveryManifestOutputs.tensorboard_logdir_candidates) ? discoveryManifestOutputs.tensorboard_logdir_candidates.join(', ') : discoveryManifestOutputs.tensorboard_logdir_candidates) }}</div>
                  </div>
                </div>
              </div>
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard title="Adapt / Start Training" :bordered="false">
              <NForm label-placement="top">
                <NFormItem label="Training selection">
                  <NSelect v-model:value="trainingRequest.selection_id" :options="selectionOptions" placeholder="Select the data batch to train" />
                </NFormItem>
                <NFormItem label="Framework profile">
                  <NSelect v-model:value="trainingRequest.framework_profile_id" :options="frameworkOptions" placeholder="Select training framework profile" />
                </NFormItem>
                <NFormItem label="Adapted dataset">
                  <NSelect
                    v-model:value="trainingRequest.dataset_run_id"
                    :options="adaptedDatasetOptions"
                    placeholder="Select a completed Adapt Data run"
                    clearable
                  />
                </NFormItem>
                <NFormItem label="Policy version">
                  <NInput v-model:value="trainingRequest.policy_version" placeholder="stack_blocks_policy_v1" />
                </NFormItem>
                <NFormItem label="Architecture">
                  <NInput v-model:value="trainingRequest.architecture" placeholder="optional" />
                </NFormItem>
                <NFormItem label="Training launch">
                  <NSelect v-model:value="trainingRequest.train_launch_mode" :options="launchModeOptions" />
                </NFormItem>
                <div v-if="trainingRequest.train_launch_mode === 'remote_tmux'" class="grid gap-2 md:grid-cols-2">
                  <NFormItem label="Remote host">
                    <NInput v-model:value="trainingRequest.remote_host" placeholder="user@training-pc" />
                  </NFormItem>
                  <NFormItem label="Remote repo root">
                    <NInput v-model:value="trainingRequest.remote_repo_root" placeholder="/home/user/Transengram_datacollection-main" />
                  </NFormItem>
                  <NFormItem label="Remote dataset dir">
                    <NInput v-model:value="trainingRequest.remote_dataset_dir" placeholder="~/robolineage_training/runs/{run_id}/dataset" />
                  </NFormItem>
                  <NFormItem label="Remote checkpoint dir">
                    <NInput v-model:value="trainingRequest.remote_checkpoint_dir" placeholder="~/robolineage_training/runs/{run_id}/checkpoints" />
                  </NFormItem>
                  <NFormItem label="SSH args">
                    <NInput v-model:value="trainingRequest.remote_ssh_args" placeholder="-p 22" />
                  </NFormItem>
                  <NFormItem label="rsync args">
                    <NInput v-model:value="trainingRequest.remote_rsync_args" placeholder="-az --delete" />
                  </NFormItem>
                </div>
                <div class="flex flex-wrap gap-2">
                  <NButton
                    type="default"
                    :disabled="!trainingRequest.selection_id || !trainingRequest.framework_profile_id || !trainingRequest.policy_version"
                    :loading="lifecycle.adaptingTrainingData"
                    @click="adaptTrainingData"
                  >
                    Adapt Data
                  </NButton>
                  <NButton
                    type="primary"
                    :disabled="startTrainingDisabled"
                    :loading="lifecycle.startingTraining"
                    @click="startTraining"
                  >
                    Start Training
                  </NButton>
                </div>
              </NForm>
              <div class="mt-4 rounded border border-gray-100 bg-gray-50 p-3 text-xs">
                <div class="mb-2 flex items-center justify-between">
                  <span class="font-medium text-gray-500">Training Monitor</span>
                  <NTag size="small" :type="training.status?.active ? 'warning' : 'default'">
                    {{ training.status?.active ? 'running' : 'idle' }}
                  </NTag>
                </div>
                <div v-if="monitorRun" class="grid gap-2 md:grid-cols-2">
                  <div>
                    <span class="text-gray-400">Run</span>
                    <div class="break-all text-gray-700">{{ monitorRun.run_id }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Status</span>
                    <div class="font-medium text-gray-700">{{ displayValue(monitorRun.status) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">tmux</span>
                    <div class="font-medium text-gray-700">{{ displayValue(monitorRun.tmux_session) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Dataset adapt</span>
                    <div class="font-medium text-gray-700">{{ displayValue(selectedAdaptDisplayStatus) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Target kind</span>
                    <div class="font-medium text-gray-700">{{ displayValue(monitorRun.target_dataset_kind) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Launch</span>
                    <div class="font-medium text-gray-700">{{ displayValue(monitorRun.launch_mode || trainingRequest.train_launch_mode) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-400">Remote</span>
                    <div class="break-all font-medium text-gray-700">{{ displayValue(monitorRun.remote_host || trainingRequest.remote_host) }}</div>
                  </div>
                  <div class="md:col-span-2">
                    <span class="text-gray-400">Remote log</span>
                    <div class="break-all text-gray-700">{{ displayValue(monitorRun.remote_train_log) }}</div>
                  </div>
                </div>
                <div v-else class="text-gray-400">No training run yet.</div>
                <div v-if="showSelectedAdaptDetail" class="mt-3 rounded border border-amber-200 bg-amber-50 p-2 text-amber-900">
                  <div v-if="selectedAdaptStatus.reason">
                    <span class="font-medium">Reason:</span> {{ displayValue(selectedAdaptStatus.reason) }}
                  </div>
                  <div v-if="selectedAdaptStatus.error">
                    <span class="font-medium">Error:</span> {{ displayValue(selectedAdaptStatus.error) }}
                  </div>
                  <ul v-if="selectedAdaptWarnings.length" class="mt-1 list-disc pl-4">
                    <li v-for="warning in selectedAdaptWarnings" :key="warning">{{ warning }}</li>
                  </ul>
                </div>
                <div v-if="hasSelectedDatasetHealth || hasSelectedTrainingUnderstanding" class="mt-3 grid gap-3 md:grid-cols-2">
                  <div v-if="hasSelectedDatasetHealth" class="rounded border border-gray-200 bg-white p-3">
                    <div class="mb-2 flex items-center justify-between gap-2">
                      <span class="font-medium text-gray-600">Dataset Health</span>
                      <NTag size="small" :type="selectedDatasetHealthUnderstanding.status === 'generated' ? 'success' : 'default'">
                        {{ displayValue(selectedDatasetHealthUnderstanding.status || selectedDatasetHealthReport.status) }}
                      </NTag>
                    </div>
                    <div class="grid gap-2 md:grid-cols-2">
                      <div>
                        <span class="text-gray-400">Action</span>
                        <div class="font-medium text-gray-700">{{ displayValue(selectedDatasetHealthReport.recommended_action) }}</div>
                      </div>
                      <div>
                        <span class="text-gray-400">Selected</span>
                        <div class="font-medium text-gray-700">{{ displayValue(selectedDatasetHealthReport.selected_rollout_count) }}</div>
                      </div>
                    </div>
                    <div v-if="selectedDatasetHealthMissingPhases.length" class="mt-2 flex flex-wrap gap-1">
                      <NTag v-for="phase in selectedDatasetHealthMissingPhases" :key="phase" size="small" type="warning">
                        {{ phase }}
                      </NTag>
                    </div>
                    <p class="mt-2 text-gray-600">
                      {{ displayValue(selectedDatasetHealthUnderstanding.summary || selectedDatasetHealthReport.summary) }}
                    </p>
                  </div>
                  <div v-if="hasSelectedTrainingUnderstanding" class="rounded border border-gray-200 bg-white p-3">
                    <div class="mb-2 flex items-center justify-between gap-2">
                      <span class="font-medium text-gray-600">Training Understanding</span>
                      <NTag size="small" :type="selectedTrainingMonitorUnderstanding.status === 'generated' ? 'success' : 'default'">
                        {{ displayValue(selectedTrainingMonitorUnderstanding.status) }}
                      </NTag>
                    </div>
                    <div>
                      <span class="text-gray-400">Action</span>
                      <div class="font-medium text-gray-700">{{ displayValue(selectedTrainingMonitorUnderstanding.recommended_action) }}</div>
                    </div>
                    <p class="mt-2 text-gray-600">
                      {{ displayValue(selectedTrainingMonitorUnderstanding.operator_brief || selectedTrainingMonitorUnderstanding.diagnosis) }}
                    </p>
                    <ul v-if="selectedTrainingLikelyCauses.length" class="mt-2 list-disc pl-4 text-gray-600">
                      <li v-for="cause in selectedTrainingLikelyCauses" :key="cause">{{ cause }}</li>
                    </ul>
                  </div>
                </div>
                <div v-if="training.selected?.train_log" class="mt-3">
                  <div class="mb-1 text-gray-400">Training log</div>
                  <pre class="max-h-44 overflow-auto rounded bg-white p-2 whitespace-pre-wrap text-gray-700">{{ training.selected.train_log }}</pre>
                </div>
                <div v-else-if="training.selected?.dataset_log" class="mt-3">
                  <div class="mb-1 text-gray-400">Dataset adapt log</div>
                  <pre class="max-h-44 overflow-auto rounded bg-white p-2 whitespace-pre-wrap text-gray-700">{{ training.selected.dataset_log }}</pre>
                </div>
              </div>
            </NCard>
          </NGridItem>
        </NGrid>

        <NCard class="mt-4" title="Framework Profiles" :bordered="false">
          <NDataTable :columns="frameworkColumns" :data="lifecycle.frameworkProfiles" :pagination="{ pageSize: 6 }" />
        </NCard>

        <NCard class="mt-4" title="Training Runs" :bordered="false">
          <NDataTable
            :columns="runColumns"
            :data="training.runs"
            :loading="training.loading"
            :pagination="{ pageSize: 8 }"
            :row-props="(row) => ({ class: 'cursor-pointer', onClick: () => { trainingRequest.dataset_run_id = row.run_id; void training.selectRun(row.run_id) } })"
          />
        </NCard>
      </NTabPane>

      <NTabPane name="policies" tab="Policies">
        <NCard title="Policy Versions" :bordered="false">
          <NDataTable :columns="policyColumns" :data="lifecycle.policies" :pagination="{ pageSize: 10 }" />
        </NCard>
      </NTabPane>

      <NTabPane name="evaluation" tab="Evaluation">
        <NGrid cols="1 xl:2" :x-gap="16" :y-gap="16" responsive="screen">
          <NGridItem>
            <NCard title="Deployment Evaluation Sessions" :bordered="false">
              <NDataTable
                :columns="deploymentSessionColumns"
                :data="lifecycle.deploymentSessions"
                :loading="lifecycle.loading || lifecycle.detailLoading"
                :pagination="{ pageSize: 8 }"
                :row-props="(row) => ({ class: 'cursor-pointer', onClick: () => selectDeploymentSession(row.session_id) })"
              />
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard title="Evaluation Analysis" :bordered="false">
              <div v-if="lifecycle.selectedDeploymentDetail" class="space-y-4">
                <div class="grid gap-2 text-sm md:grid-cols-2">
                  <div>
                    <span class="text-gray-500">Session</span>
                    <div class="font-medium">{{ displayValue(selectedDeploymentSession.session_id) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-500">Policy</span>
                    <div class="font-medium">{{ displayValue(selectedDeploymentSession.policy_version) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-500">Success rate</span>
                    <div class="font-medium">{{ displayValue(selectedDeploymentSummary.success_rate) }}</div>
                  </div>
                  <div>
                    <span class="text-gray-500">Decision</span>
                    <div>
                      <NTag size="small" :type="decisionTagType(selectedDeploymentDecision.decision)">
                        {{ displayValue(selectedDeploymentDecision.decision) }}
                      </NTag>
                    </div>
                  </div>
                </div>

                <div class="space-y-2">
                  <p class="text-xs text-gray-400">Collection recommendation</p>
                  <div class="flex flex-wrap gap-2">
                    <NTag size="small" type="info">
                      {{ displayValue(selectedCollectionRecommendation.recommended_mode) }}
                    </NTag>
                    <NTag
                      v-for="phase in selectedFocusPhases"
                      :key="phase"
                      size="small"
                      type="warning"
                    >
                      {{ phase }}
                    </NTag>
                  </div>
                  <p class="text-sm text-gray-600">
                    {{ displayValue(selectedNextCollectionBrief.operator_brief) }}
                  </p>
                </div>

                <div v-if="hasSelectedGovernanceUnderstanding" class="rounded border border-gray-200 bg-gray-50 p-3 text-sm">
                  <div class="mb-2 flex items-center justify-between gap-2">
                    <span class="font-medium text-gray-600">Governance Understanding</span>
                    <NTag size="small" :type="selectedGovernanceUnderstanding.status === 'generated' ? 'success' : 'default'">
                      {{ displayValue(selectedGovernanceUnderstanding.status) }}
                    </NTag>
                  </div>
                  <p class="text-gray-600">
                    {{ displayValue(selectedGovernanceUnderstanding.operator_brief || selectedGovernanceUnderstanding.summary) }}
                  </p>
                  <div class="mt-3 grid gap-2 md:grid-cols-3">
                    <div>
                      <span class="text-xs text-gray-400">Rule decision</span>
                      <div class="font-medium text-gray-700">{{ displayValue(selectedGovernanceUnderstanding.deterministic_decision) }}</div>
                    </div>
                    <div>
                      <span class="text-xs text-gray-400">LLM suggestion</span>
                      <div>
                        <NTag size="small" :type="decisionTagType(selectedGovernanceUnderstanding.llm_suggested_decision)">
                          {{ displayValue(selectedGovernanceUnderstanding.llm_suggested_decision) }}
                        </NTag>
                      </div>
                    </div>
                    <div>
                      <span class="text-xs text-gray-400">Confidence</span>
                      <div class="font-medium text-gray-700">{{ displayValue(selectedGovernanceUnderstanding.confidence) }}</div>
                    </div>
                  </div>
                  <ul v-if="selectedGovernanceRiskNotes.length" class="mt-2 list-disc pl-4 text-gray-600">
                    <li v-for="note in selectedGovernanceRiskNotes" :key="note">{{ note }}</li>
                  </ul>
                </div>

                <div>
                  <p class="mb-2 text-xs text-gray-400">Eval rollouts</p>
                  <NDataTable
                    :columns="evalRolloutColumns"
                    :data="selectedDeploymentRollouts"
                    :pagination="{ pageSize: 6 }"
                    :row-props="(row) => ({ class: 'cursor-pointer', onClick: () => selectEvalRollout(row) })"
                  />
                </div>

                <div v-if="selectedEvalRollout" class="space-y-2">
                  <div class="flex flex-wrap gap-2">
                    <NTag size="small" type="info">{{ displayValue(selectedEvalRollout.rollout_id) }}</NTag>
                    <NTag size="small" :type="selectedEvalRollout.final_success === true ? 'success' : selectedEvalRollout.final_success === false ? 'error' : 'default'">
                      success: {{ displayValue(selectedEvalRollout.final_success) }}
                    </NTag>
                    <NTag size="small" type="warning">
                      {{ displayValue(selectedEvalRollout.recommended_next_action) }}
                    </NTag>
                  </div>
                  <p class="text-sm text-gray-600">{{ displayValue(selectedEvalRollout.policy_behavior_summary) }}</p>
                  <pre class="max-h-56 overflow-auto rounded bg-gray-50 p-3 text-xs whitespace-pre-wrap">{{ selectedEvalRollout.eval_review_report || 'No rollout report yet.' }}</pre>
                </div>

                <div>
                  <p class="mb-2 text-xs text-gray-400">Session report</p>
                  <pre class="max-h-56 overflow-auto rounded bg-gray-50 p-3 text-xs whitespace-pre-wrap">{{ lifecycle.selectedDeploymentDetail.deployment_session_report || 'No session report yet.' }}</pre>
                </div>
              </div>
              <div v-else class="text-sm text-gray-500">Select a deployment evaluation session on the left to inspect policy evaluation, deployment decision, and recollection guidance.</div>
            </NCard>
          </NGridItem>
        </NGrid>
      </NTabPane>
    </NTabs>
  </div>
</template>
