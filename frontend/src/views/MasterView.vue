<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NCode,
  NDescriptions,
  NDescriptionsItem,
  NSpace,
  NTag,
  NTimeline,
  NTimelineItem,
} from 'naive-ui'
import { useMasterStore } from '@/stores/master'

const store = useMasterStore()
let refreshTimer: number | null = null
onMounted(() => {
  void store.refresh()
  refreshTimer = window.setInterval(() => {
    if (!store.reviewing) void store.refresh()
  }, 5000)
})
onUnmounted(() => {
  if (refreshTimer !== null) {
    window.clearInterval(refreshTimer)
    refreshTimer = null
  }
})

const state = computed(() => store.status?.state ?? {})
const review = computed(() => store.status?.review ?? {})
const understanding = computed(() => store.status?.understanding ?? {})
const nextAction = computed(() => asRecord(state.value.next_action ?? review.value.next_action))
const risks = computed(() => {
  const value = state.value.risks
  return Array.isArray(value) ? value.map((item) => asRecord(item)) : []
})
const blocking = computed(() => {
  const value = state.value.blocking
  return Array.isArray(value) ? value.length : 0
})
const events = computed(() => (store.status?.events ?? [])
  .slice(-10)
  .reverse()
  .map((item) => {
    const event = asRecord(item)
    return {
      name: displayValue(event.event, 'event'),
      createdAt: displayValue(event.created_at),
      payload: event.payload,
    }
  }))
const memory = computed(() => store.status?.memory ?? [])
const stage = computed(() => displayValue(state.value.current_stage, 'unknown'))
const action = computed(() => displayValue(nextAction.value.action, 'unknown'))
const reason = computed(() => displayValue(nextAction.value.reason))
const healthStatus = computed(() => displayValue(asRecord(store.status?.health_summary).status, 'unknown'))
const understandingStatus = computed(() => displayValue(understanding.value.status, 'not_configured'))
const understandingSummary = computed(() => displayValue(understanding.value.operator_brief || understanding.value.summary))
const lastReviewTrigger = computed(() => displayValue(store.status?.last_review_trigger, 'none'))
const lastReviewAt = computed(() => displayValue(store.status?.last_review_at))
const lastReviewMode = computed(() => lastReviewTrigger.value === 'manual' ? 'manual' : lastReviewTrigger.value === 'none' ? 'none' : 'auto')
const understandingRisks = computed(() => {
  const value = understanding.value.risk_interpretation
  return Array.isArray(value) ? value.map((item) => asRecord(item)) : []
})
const llmSuggestedNextAction = computed(() => asRecord(understanding.value.suggested_next_action))
const rawStatus = computed(() => JSON.stringify(store.status, null, 2))

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function displayValue(value: unknown, fallback = '-'): string {
  if (value === null || value === undefined || value === '') return fallback
  return String(value)
}

function stageTagType(value: string) {
  if (value === 'deployment_governance') return 'success'
  if (value === 'training') return 'info'
  if (value === 'not_started') return 'default'
  return 'warning'
}

function riskTagType(value: unknown) {
  const severity = String(value ?? '').toLowerCase()
  if (severity === 'high') return 'error'
  if (severity === 'medium') return 'warning'
  return 'info'
}
</script>

<template>
  <div class="space-y-4">
    <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h2 class="text-2xl font-semibold">Master</h2>
        <p class="mt-1 text-sm text-gray-500">Global lifecycle state, memory, and checkpoint review.</p>
      </div>
      <NSpace>
        <NButton :loading="store.loading" @click="store.refresh()">
          Refresh
        </NButton>
        <NButton type="primary" :loading="store.reviewing" @click="store.runReview()">
          Run Master Review
        </NButton>
      </NSpace>
    </div>

    <NAlert v-if="store.lastError" type="error" :title="store.lastError" />
    <NAlert v-if="store.status?.last_error" type="error" :title="store.status.last_error" />
    <NAlert v-if="store.status && !store.status.available" type="info" title="No Master state yet" />

    <div class="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
      <NCard title="State">
        <NDescriptions :column="1" bordered size="small">
          <NDescriptionsItem label="Task">
            {{ displayValue(state.task_id) }}
          </NDescriptionsItem>
          <NDescriptionsItem label="Stage">
            <NTag :type="stageTagType(stage)" size="small">{{ stage }}</NTag>
          </NDescriptionsItem>
          <NDescriptionsItem label="Next action">
            <div class="space-y-1">
              <div class="font-medium">{{ action }}</div>
              <div class="text-xs text-gray-500">{{ reason }}</div>
            </div>
          </NDescriptionsItem>
          <NDescriptionsItem label="Health">
            {{ healthStatus }}
          </NDescriptionsItem>
          <NDescriptionsItem label="Last review">
            <div class="space-y-1">
              <div class="flex items-center gap-2">
                <NTag :type="lastReviewMode === 'auto' ? 'success' : lastReviewMode === 'manual' ? 'info' : 'default'" size="small">
                  {{ lastReviewMode }}
                </NTag>
                <span>{{ lastReviewTrigger }}</span>
              </div>
              <div class="text-xs text-gray-500">{{ lastReviewAt }}</div>
            </div>
          </NDescriptionsItem>
          <NDescriptionsItem label="Blocking risks">
            {{ blocking }}
          </NDescriptionsItem>
          <NDescriptionsItem label="Task root">
            <span class="break-all text-xs">{{ displayValue(store.status?.task_root) }}</span>
          </NDescriptionsItem>
        </NDescriptions>
      </NCard>

      <NCard title="Risks">
        <div v-if="risks.length" class="space-y-3">
          <div v-for="risk in risks" :key="displayValue(risk.code)" class="space-y-1 rounded border border-gray-200 p-3">
            <div class="flex items-center justify-between gap-3">
              <span class="font-medium">{{ displayValue(risk.code) }}</span>
              <NTag :type="riskTagType(risk.severity)" size="small">
                {{ displayValue(risk.severity) }}
              </NTag>
            </div>
            <div class="text-sm text-gray-500">{{ displayValue(risk.message) }}</div>
          </div>
        </div>
        <div v-else class="text-sm text-gray-500">No risks recorded.</div>
      </NCard>
    </div>

    <NCard title="LLM Understanding">
      <div v-if="store.status?.understanding" class="space-y-4">
        <NDescriptions :column="1" bordered size="small">
          <NDescriptionsItem label="Status">
            <NTag :type="understandingStatus === 'generated' ? 'success' : understandingStatus === 'failed' ? 'error' : 'warning'" size="small">
              {{ understandingStatus }}
            </NTag>
          </NDescriptionsItem>
          <NDescriptionsItem label="Model">
            {{ displayValue(understanding.model) }}
          </NDescriptionsItem>
          <NDescriptionsItem label="Brief">
            {{ understandingSummary }}
          </NDescriptionsItem>
          <NDescriptionsItem label="LLM next action">
            <div class="space-y-1">
              <div class="font-medium">{{ displayValue(llmSuggestedNextAction.action) }}</div>
              <div class="text-xs text-gray-500">{{ displayValue(llmSuggestedNextAction.reason) }}</div>
            </div>
          </NDescriptionsItem>
        </NDescriptions>
        <div v-if="understandingRisks.length" class="space-y-2">
          <div v-for="risk in understandingRisks" :key="displayValue(risk.code)" class="rounded border border-gray-200 p-3 text-sm">
            <div class="font-medium">{{ displayValue(risk.code) }}</div>
            <div class="text-gray-500">{{ displayValue(risk.reason || risk.message) }}</div>
          </div>
        </div>
      </div>
      <div v-else class="text-sm text-gray-500">No LLM understanding recorded.</div>
    </NCard>

    <div class="grid gap-4 xl:grid-cols-2">
      <NCard title="Recent Events">
        <NTimeline v-if="events.length">
          <NTimelineItem
            v-for="event in events"
            :key="`${event.name}-${event.createdAt}`"
            :title="event.name"
            :content="JSON.stringify(event.payload ?? {})"
            :time="event.createdAt"
          />
        </NTimeline>
        <div v-else class="text-sm text-gray-500">No events recorded.</div>
      </NCard>

      <NCard title="Memory">
        <NCode :code="JSON.stringify(memory, null, 2)" language="json" />
      </NCard>
    </div>

    <NCard title="Raw Master Status">
      <NCode :code="rawStatus" language="json" />
    </NCard>
  </div>
</template>
