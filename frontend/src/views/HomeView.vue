<script setup lang="ts">
import { computed, h, onMounted, onUnmounted, ref } from 'vue'
import { NAlert, NButton, NCard, NDescriptions, NDescriptionsItem, NInput, NModal, NTag, NList, NListItem, NDataTable, NScrollbar, NSelect, NRadioButton, NRadioGroup } from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { useLifecycleStore } from '@/stores/lifecycle'
import { useTaskStore } from '@/stores/task'

const store = useTaskStore()
const lifecycleStore = useLifecycleStore()

interface ParsedFull {
  phase: string
  progress: string
  risk_level: string
  imminent_failure: boolean
  needs_review: boolean
  confidence: number
}

interface Prior {
  top_phase: string
  top_margin: number
  phase_scores: Record<string, number>
  prior_reason: string
}

interface Decision {
  rollout_id?: string
  rollout_index?: number | null
  timestamp: string
  phase: string
  progress: string
  event_type: string
  anchor_frame: number | null
  end_frame: number | null
  n_images: number
  image_paths: string[]
  prior: Prior
  parsed_full: ParsedFull
  raw_response: string
}

type SessionMode = 'collection' | 'deployment'

interface TaskConfigPreview {
  task_description?: string
  phases?: string[]
  phase_definitions?: Record<string, string>
  phase_action_hints?: Record<string, unknown>
  phase_visual_hints?: Record<string, unknown>
  failure_signals?: string[]
}

const decisions = ref<Decision[]>([])
const showModal = ref(false)
const selectedRow = ref<Decision | null>(null)
const deploymentPolicyVersion = ref('')
const sessionMode = ref<SessionMode>('collection')
const activeTask = computed(() => lifecycleStore.tasks.find((item) => item.task_id === lifecycleStore.activeTaskId) ?? null)
const hasTaskConfig = computed(() => Boolean(store.result?.task_config_path || store.rollout.task_config_path || activeTask.value?.latest_task_config_path))
const canUpdateTaskConfig = computed(() => !store.rollout.active && (!store.rolloutSession.active || store.rolloutSession.rollout_count === 0))
const activeSessionMode = computed<SessionMode>(() => (store.rolloutSession.kind === 'deployment' ? 'deployment' : store.rolloutSession.kind === 'collection' ? 'collection' : sessionMode.value))
const primaryTaskConfigLabel = computed(() => lifecycleStore.activeTaskId ? 'Update Current Task Config' : 'Create Task and Generate Task Config')
const taskConfigPreview = computed<TaskConfigPreview | null>(() => {
  if (store.result) {
    return {
      task_description: store.result.task_description,
      phases: store.result.phases,
      phase_definitions: store.result.phase_definitions,
      phase_action_hints: store.result.phase_action_hints,
      phase_visual_hints: store.result.phase_visual_hints,
      failure_signals: store.result.failure_signals,
    }
  }
  const cfg = activeTask.value?.latest_task_config
  return cfg ? (cfg as TaskConfigPreview) : null
})
const taskConfigVersion = computed(() => store.result?.task_config_version || activeTask.value?.latest_task_config_version || null)
const taskConfigPath = computed(() => store.result?.task_config_path || activeTask.value?.latest_task_config_path || store.rollout.task_config_path || null)
const phaseRows = computed(() => {
  const cfg = taskConfigPreview.value
  return (cfg?.phases ?? []).map((phase, index) => ({
    phase,
    index,
    definition: cfg?.phase_definitions?.[phase] ?? '',
    visual: displayValue((cfg?.phase_visual_hints ?? {})[phase]),
    action: hintSummary((cfg?.phase_action_hints ?? {})[phase]),
  }))
})
const collectionAnalysisHref = computed(() => {
  const summary = store.lastSessionSummary
  const sessionId = summary && !summary.deployment_decision ? String(summary.session_id ?? '') : ''
  return `/review-lifecycle?tab=review${sessionId ? `&session=${encodeURIComponent(sessionId)}` : ''}`
})
const evaluationAnalysisHref = computed(() => {
  const summary = store.lastSessionSummary
  const sessionId = summary?.deployment_decision ? String(summary.session_id ?? '') : ''
  return `/review-lifecycle?tab=evaluation${sessionId ? `&session=${encodeURIComponent(sessionId)}` : ''}`
})

function openDetail(row: Decision) {
  selectedRow.value = row
  showModal.value = true
}

function imageUrl(path: string) {
  return `/api/session/vsa/rollout-image?path=${encodeURIComponent(path)}`
}

const columns: DataTableColumns<Decision> = [
  {
    title: 'Rollout',
    key: 'rollout_index',
    width: 90,
    render(row) {
      return row.rollout_index ? `#${row.rollout_index}` : (row.rollout_id ?? '—')
    },
  },
  {
    title: 'Time',
    key: 'timestamp',
    width: 180,
    ellipsis: true,
  },
  {
    title: 'phase',
    key: 'phase',
    width: 200,
    ellipsis: true,
  },
  {
    title: 'progress',
    key: 'progress',
    ellipsis: true,
  },
  {
    title: 'Action',
    key: 'actions',
    width: 70,
    render(row) {
      return h(NButton, { size: 'small', onClick: () => openDetail(row) }, { default: () => 'View' })
    },
  },
]

function displayValue(value: unknown) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

function hintSummary(value: unknown) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(displayValue).join(', ')
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${key}: ${Array.isArray(item) ? item.join(', ') : displayValue(item)}`)
      .join('; ')
  }
  return String(value)
}

const taskOptions = computed(() => lifecycleStore.tasks.map((item) => ({
  label: `${item.display_name || item.task_id} (${item.rollout_count} rollout)`,
  value: item.task_id,
})))

const activeTaskName = computed(() => {
  const task = lifecycleStore.tasks.find((item) => item.task_id === lifecycleStore.activeTaskId)
  return task?.display_name || lifecycleStore.activeTaskId || 'Not selected'
})

function sessionStatusText() {
  const session = store.rolloutSession
  if (!session.active) return 'Not started'
  const kind = session.kind === 'deployment' ? 'deployment evaluation' : 'collection'
  if (session.finalizing) {
    const stage = session.finalization_stage ? ` / ${session.finalization_stage}` : ''
    return `${kind}  session closing: ${session.rollout_count} rollouts${stage}`
  }
  return `${kind} session: ${session.rollout_count} rollouts`
}

function sessionTagType(): 'success' | 'warning' | 'error' | 'info' | 'default' {
  if (store.rolloutSession.status === 'finalization_failed') return 'error'
  if (store.rolloutSession.finalizing) return 'info'
  if (!store.rolloutSession.active) return 'default'
  return store.rolloutSession.kind === 'deployment' ? 'warning' : 'success'
}

function lastSessionSummaryText() {
  const summary = store.lastSessionSummary
  if (!summary) return ''
  const deployment = summary.deployment_decision as Record<string, unknown> | undefined
  const collection = summary.collection_recommendation as Record<string, unknown> | undefined
  const brief = summary.next_collection_brief as Record<string, unknown> | undefined
  if (deployment) {
    return `Latest deployment decision: ${displayValue(deployment.decision)}; recollection guidance: ${displayValue(collection?.recommended_mode)}; ${displayValue(brief?.operator_brief)}`
  }
  return `Latest collection session: ${displayValue(summary.rollout_count)} rollouts; output: ${displayValue(summary.output_dir)}`
}

let pollTimer: ReturnType<typeof setInterval> | null = null

async function fetchDecisions() {
  try {
    const resp = await fetch('/api/session/vsa/decisions?n=50')
    if (resp.ok) {
      const data = await resp.json()
      decisions.value = (data as Decision[]).reverse()
    }
  } catch {
    // silently ignore — backend may not be running
  }
}

async function startRollout() {
  await store.startRollout()
  await fetchDecisions()
}

async function stopRollout() {
  await store.stopRollout()
}

async function startCollectionSession() {
  await store.startCollectionSession()
}

async function stopCollectionSession() {
  await store.stopCollectionSession()
}

async function startDeploymentSession() {
  await store.startDeploymentSession(deploymentPolicyVersion.value)
}

async function stopDeploymentSession() {
  await store.stopDeploymentSession()
}

async function activateLifecycleTask(taskId: string) {
  await lifecycleStore.activateTask(taskId)
  const active = lifecycleStore.tasks.find((item) => item.task_id === lifecycleStore.activeTaskId)
  if (active?.task_description) store.description = active.task_description
  store.result = null
  store.lastSessionSummary = null
  await store.refreshRollout()
  await store.refreshRolloutSession()
  await fetchDecisions()
}

async function createTaskAndGenerate() {
  const description = store.description.trim()
  if (!description) return
  const task = await lifecycleStore.createTask({
    task_description: description,
    display_name: description,
  })
  if (!task) return
  await store.generate()
  await lifecycleStore.refreshTasks()
}

async function generateCurrentTaskConfig() {
  if (!lifecycleStore.activeTaskId) {
    await createTaskAndGenerate()
    return
  }
  await store.generate()
  await lifecycleStore.refreshTasks()
}

async function runPrimaryTaskConfigAction() {
  if (lifecycleStore.activeTaskId) {
    await generateCurrentTaskConfig()
  } else {
    await createTaskAndGenerate()
  }
}

onMounted(() => {
  lifecycleStore.refreshTasks()
  store.refreshRollout()
  store.refreshRolloutSession()
  fetchDecisions()
  pollTimer = setInterval(() => {
    fetchDecisions()
    store.refreshRolloutSession()
  }, 3000)
})

onUnmounted(() => {
  if (pollTimer !== null) clearInterval(pollTimer)
})
</script>

<template>
  <div class="h-full space-y-6">
    <div>
      <h2 class="text-2xl font-semibold">Rollout Control</h2>
      <p class="mt-1 text-sm text-gray-500">Create task phase configuration and organize collection or evaluation rollouts by session.</p>
    </div>

    <div class="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(420px,1.05fr)]">
      <div class="min-w-0 space-y-6">
      <NCard title="Task Setup">
        <div class="space-y-3">
          <NInput
            v-model:value="store.description"
            type="textarea"
            :rows="4"
            placeholder="Example: pick up the red block and place it on the tray"
            :disabled="store.loading"
            @keydown.ctrl.enter="generateCurrentTaskConfig"
          />
          <div class="rounded border border-gray-200 bg-gray-50 p-3 space-y-2">
            <div class="flex items-center justify-between gap-3">
              <div class="min-w-0">
                <div class="flex items-center gap-2">
                  <span class="text-sm font-medium">Current task</span>
                  <NTag size="small" type="info">{{ activeTaskName }}</NTag>
                </div>
                <p class="mt-1 text-xs text-gray-400">
                  All rollouts, sessions, reviews, training selections, and policies are filed under the current task.
                </p>
              </div>
              <NButton size="small" :loading="lifecycleStore.loading" @click="lifecycleStore.refreshTasks()">
                Refresh History
              </NButton>
            </div>
            <div class="flex flex-wrap gap-2">
              <NSelect
                v-model:value="lifecycleStore.activeTaskId"
                :options="taskOptions"
                size="small"
                filterable
                placeholder="Select previous task"
                style="min-width: 260px; flex: 1"
                :disabled="store.rollout.active || store.rolloutSession.active"
                @update:value="value => value && activateLifecycleTask(String(value))"
              />
              <NButton
                size="small"
                type="default"
                :loading="store.loading || lifecycleStore.activatingTask"
                :disabled="!store.description.trim() || store.rollout.active || store.rolloutSession.active"
                @click="createTaskAndGenerate"
              >
                Create a new task and generate
              </NButton>
            </div>
          </div>
          <div class="flex gap-2">
            <NButton
              type="primary"
              :loading="store.loading"
              :disabled="!store.description.trim() || store.starting || store.stopping || !canUpdateTaskConfig"
              @click="runPrimaryTaskConfigAction"
            >
              {{ primaryTaskConfigLabel }}
            </NButton>
            <a
              href="/review-lifecycle"
              class="inline-flex items-center rounded border border-gray-200 px-3 text-sm text-gray-600 hover:border-blue-300 hover:text-blue-600"
            >
              View Review / Lifecycle
            </a>
          </div>
          <p class="text-xs text-gray-400">
            Task Config provides shared phase semantics for online VSA, post-rollout review, and evaluation.
          </p>
        </div>
      </NCard>

      <NCard title="Task Config Preview">
        <div v-if="taskConfigPreview" class="space-y-4">
          <NDescriptions bordered size="small" :column="1">
            <NDescriptionsItem label="version">{{ displayValue(taskConfigVersion) }}</NDescriptionsItem>
            <NDescriptionsItem label="path">{{ displayValue(taskConfigPath) }}</NDescriptionsItem>
            <NDescriptionsItem label="task_description">{{ displayValue(taskConfigPreview.task_description) }}</NDescriptionsItem>
          </NDescriptions>

          <div>
            <p class="mb-2 text-xs text-gray-400">Phases / hints</p>
            <NList bordered size="small">
              <NListItem v-for="row in phaseRows" :key="row.phase">
                <div class="space-y-1">
                  <div class="flex items-center gap-2">
                    <NTag type="info" size="small">{{ row.index + 1 }}</NTag>
                    <span class="font-mono text-xs text-blue-600">{{ row.phase }}</span>
                  </div>
                  <p class="text-sm text-gray-600">{{ displayValue(row.definition) }}</p>
                  <p class="text-xs text-gray-500">visual: {{ row.visual }}</p>
                  <p class="text-xs text-gray-500">action: {{ row.action }}</p>
                </div>
              </NListItem>
            </NList>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs text-gray-400">Failure signals</span>
            <NTag
              v-for="sig in taskConfigPreview.failure_signals || []"
              :key="sig"
              type="warning"
              size="small"
            >
              {{ sig }}
            </NTag>
            <span v-if="!(taskConfigPreview.failure_signals || []).length" class="text-xs text-gray-400">None</span>
          </div>
        </div>
        <div v-else class="text-sm text-gray-500">
          The current task has no Task Config. Generate it before VSA, post-review, and evaluation.
        </div>
      </NCard>

      <NAlert v-if="store.error" type="error" :title="store.error" />
    </div>

    <div class="min-w-0 space-y-6">
      <NCard title="Session Setting">
        <div class="space-y-3">
          <div class="flex items-center justify-between gap-3">
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium">Mode</span>
                <NTag size="small" :type="sessionTagType()">
                  {{ sessionStatusText() }}
                </NTag>
              </div>
              <p class="mt-1 text-xs text-gray-400">
                Choose a session type before running rollouts; the type is locked within an active session.
              </p>
            </div>
            <span class="text-xs text-gray-400">{{ store.rolloutSession.session_id ?? 'no active session' }}</span>
          </div>

          <NRadioGroup v-model:value="sessionMode" :disabled="store.rolloutSession.active || store.rollout.active" size="small">
            <NRadioButton value="collection">Collection</NRadioButton>
            <NRadioButton value="deployment">Evaluation</NRadioButton>
          </NRadioGroup>

          <div v-if="activeSessionMode === 'deployment'" class="space-y-2">
            <NInput
              v-model:value="deploymentPolicyVersion"
              size="small"
              placeholder="policy version, optional"
              :disabled="store.rolloutSession.active"
            />
            <p class="text-xs text-gray-400">
              Evaluation sessions feed PolicyEvaluationAgent and DeploymentGovernanceAgent; this does not switch a policy server.
            </p>
          </div>

          <p v-else class="text-xs text-gray-400">
            Collection sessions organize multiple data rollouts under one task; each closed rollout enters post-review and dataset admission.
          </p>

          <div class="flex flex-wrap items-center gap-2">
            <template v-if="activeSessionMode === 'collection'">
              <NButton
                size="small"
                type="success"
                :loading="store.startingCollection"
                :disabled="store.rolloutSession.active || store.rollout.active || !hasTaskConfig"
                @click="startCollectionSession"
              >
                Start Collection
              </NButton>
              <NButton
                size="small"
                type="error"
                :loading="store.stoppingCollection"
                :disabled="store.rolloutSession.kind !== 'collection' || store.rolloutSession.finalizing"
                @click="stopCollectionSession"
              >
                Stop Collection
              </NButton>
              <a
                :href="collectionAnalysisHref"
                class="inline-flex items-center rounded border border-gray-200 px-3 py-1 text-sm text-gray-600 hover:border-blue-300 hover:text-blue-600"
              >
                View Post Review
              </a>
            </template>
            <template v-else>
              <NButton
                size="small"
                type="warning"
                :loading="store.startingDeployment"
                :disabled="store.rolloutSession.active || store.rollout.active || !hasTaskConfig"
                @click="startDeploymentSession"
              >
                Start Evaluation
              </NButton>
              <NButton
                size="small"
                type="error"
                :loading="store.stoppingDeployment"
                :disabled="store.rolloutSession.kind !== 'deployment' || store.rolloutSession.finalizing"
                @click="stopDeploymentSession"
              >
                Stop Evaluation
              </NButton>
              <a
                :href="evaluationAnalysisHref"
                class="inline-flex items-center rounded border border-gray-200 px-3 py-1 text-sm text-gray-600 hover:border-blue-300 hover:text-blue-600"
              >
                View Evaluation Analysis
              </a>
            </template>
          </div>

          <p v-if="store.lastSessionSummary" class="text-xs text-gray-500">
            {{ lastSessionSummaryText() }}
          </p>
          <p v-if="!hasTaskConfig" class="text-xs text-amber-600">Generate Task Config before starting a session.</p>
        </div>
      </NCard>

      <NCard title="Single Rollout">
        <div class="space-y-3">
          <div class="flex items-center justify-between gap-3">
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <span class="text-sm font-medium">Current rollout</span>
                <NTag size="small" :type="store.rollout.active ? 'success' : 'default'">
                  {{ store.rollout.rollout_id ?? 'Not started' }}
                </NTag>
                <NTag size="small" :type="sessionTagType()">
                  {{ store.rolloutSession.kind ?? 'no session' }}
                </NTag>
              </div>
              <p class="mt-1 text-xs text-gray-400">Start a session before starting a rollout; after stop, the rollout enters the matching offline analysis.</p>
            </div>
            <span v-if="store.rollout.active" class="text-xs text-green-600">Collecting</span>
            <span v-else-if="store.rolloutSession.finalizing" class="text-xs text-blue-600">
              Session is closing; waiting for VSA and review
            </span>
            <span v-else-if="store.rollout.analysis_draining" class="text-xs text-amber-600">
              VSA background analysis {{ store.rollout.draining_rollouts?.length ?? 0 }}
            </span>
          </div>

          <div class="flex flex-wrap gap-2">
            <NButton
              type="success"
              :loading="store.starting"
              :disabled="store.loading || store.stopping || store.rollout.active || !store.rolloutSession.accepting_rollouts || !hasTaskConfig"
              @click="startRollout"
            >
              Start Rollout
            </NButton>
            <NButton
              type="error"
              :loading="store.stopping"
              :disabled="store.loading || store.starting || !store.rollout.active"
              @click="stopRollout"
            >
              Stop Rollout
            </NButton>
          </div>

          <p v-if="store.rolloutSession.finalizing" class="text-xs text-blue-600">
            Current session is closing: remaining VSA and post-review jobs will finish before summary and training selection.
          </p>
          <p v-else-if="!store.rolloutSession.active" class="text-xs text-amber-600">No active session; rollout is disabled.</p>
          <p v-else-if="store.rolloutSession.active && store.rolloutSession.rollout_count > 0" class="text-xs text-gray-400">
            This session already has rollouts; Task Config is locked for this batch.
          </p>
        </div>
      </NCard>

      <NCard title="VSA Phase Decisions">
        <NDataTable
          :columns="columns"
          :data="decisions"
          :max-height="320"
          :scroll-x="650"
          size="small"
          striped
        />
      </NCard>
    </div>
  </div>
  </div>

  <!-- Inference Details Modal -->
  <NModal v-model:show="showModal" preset="card" title="Inference Details" style="width: 760px; max-width: 95vw">
    <NScrollbar style="max-height: 70vh" v-if="selectedRow">
      <div class="space-y-5 pr-2">

        <!-- Input Images -->
        <div>
          <p class="text-xs text-gray-400 mb-2">Input Images ({{ selectedRow.n_images }} images)</p>
          <div class="flex gap-2 flex-wrap">
            <img
              v-for="(p, i) in selectedRow.image_paths"
              :key="i"
              :src="imageUrl(p)"
              class="h-40 rounded border object-cover"
              :alt="`frame ${i}`"
            />
            <p v-if="!selectedRow.image_paths.length" class="text-sm text-gray-400">No images</p>
          </div>
        </div>

        <!-- Decision Information -->
        <div>
          <p class="text-xs text-gray-400 mb-2">Decision Information</p>
          <NDescriptions bordered size="small" :column="2">
            <NDescriptionsItem label="timestamp">{{ selectedRow.timestamp }}</NDescriptionsItem>
            <NDescriptionsItem label="event_type">{{ selectedRow.event_type }}</NDescriptionsItem>
            <NDescriptionsItem label="anchor_frame">{{ selectedRow.anchor_frame }}</NDescriptionsItem>
            <NDescriptionsItem label="end_frame">{{ selectedRow.end_frame }}</NDescriptionsItem>
            <NDescriptionsItem label="phase">{{ selectedRow.phase }}</NDescriptionsItem>
            <NDescriptionsItem label="progress">{{ selectedRow.progress }}</NDescriptionsItem>
          </NDescriptions>
        </div>

        <!-- Parsed Result -->
        <div>
          <p class="text-xs text-gray-400 mb-2">VLM Parsed Result</p>
          <NDescriptions bordered size="small" :column="2">
            <NDescriptionsItem label="confidence">{{ selectedRow.parsed_full.confidence }}</NDescriptionsItem>
            <NDescriptionsItem label="risk_level">{{ selectedRow.parsed_full.risk_level }}</NDescriptionsItem>
            <NDescriptionsItem label="imminent_failure">{{ selectedRow.parsed_full.imminent_failure }}</NDescriptionsItem>
            <NDescriptionsItem label="needs_review">{{ selectedRow.parsed_full.needs_review }}</NDescriptionsItem>
          </NDescriptions>
        </div>

        <!-- Action Prior -->
        <div>
          <p class="text-xs text-gray-400 mb-2">Action Prior (Prior)</p>
          <NDescriptions bordered size="small" :column="2">
            <NDescriptionsItem label="top_phase">{{ selectedRow.prior.top_phase }}</NDescriptionsItem>
            <NDescriptionsItem label="top_margin">{{ selectedRow.prior.top_margin }}</NDescriptionsItem>
            <NDescriptionsItem label="prior_reason">{{ selectedRow.prior.prior_reason }}</NDescriptionsItem>
          </NDescriptions>
          <div class="mt-2 space-y-1">
            <div
              v-for="(score, phaseName) in selectedRow.prior.phase_scores"
              :key="phaseName"
              class="flex items-center gap-2 text-xs"
            >
              <span class="w-40 truncate text-gray-600 font-mono">{{ phaseName }}</span>
              <div class="flex-1 bg-gray-100 rounded h-3 overflow-hidden">
                <div class="h-full bg-blue-400 rounded" :style="{ width: `${(score * 100).toFixed(0)}%` }" />
              </div>
              <span class="w-10 text-right text-gray-500">{{ (score * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>

        <!-- Raw VLM Response -->
        <div>
          <p class="text-xs text-gray-400 mb-2">Raw VLM Response</p>
          <pre class="text-xs bg-gray-50 border rounded p-3 whitespace-pre-wrap break-all">{{ selectedRow.raw_response }}</pre>
        </div>

      </div>
    </NScrollbar>
  </NModal>
</template>
