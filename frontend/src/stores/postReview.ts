import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface VlmUsage {
  online_active: boolean
  online_rollout_id: string | null
  offline_waiting: number
  offline_inflight: number
}

export interface PostReviewStatus {
  active: boolean
  queue_size: number
  current_rollout?: string | null
  queued_rollouts?: string[]
  last_rollout?: string | null
  last_error?: string | null
  vlm_usage?: VlmUsage
}

export interface ReviewRollout {
  rollout_id: string
  rollout_dir: string
  status: string
  success_likely: boolean | null
  dataset_decision: string | null
  accepted_for_training: boolean | null
  requires_review: boolean | null
  admission_class: string | null
  label_quality: string | null
  final_phase: string | null
  snapshot_count: number | null
  needs_review_count: number | null
  failure_candidate_count: number | null
  updated_at: string | null
}

export interface ReviewDetail {
  rollout: ReviewRollout
  evidence_index: Record<string, unknown> | null
  annotation: Record<string, unknown> | null
  rollout_summary: Record<string, unknown> | null
  failure_analysis: Record<string, unknown> | null
  dataset_admission: Record<string, unknown> | null
  post_review_status: Record<string, unknown> | null
  phase_timeline: Record<string, unknown>[]
  review_report: string
}

export const usePostReviewStore = defineStore('postReview', () => {
  const status = ref<PostReviewStatus | null>(null)
  const rollouts = ref<ReviewRollout[]>([])
  const selected = ref<ReviewDetail | null>(null)
  const loading = ref(false)
  const detailLoading = ref(false)
  const error = ref<string | null>(null)

  async function refresh(limit = 80) {
    loading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/post-review/rollouts?limit=${limit}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      status.value = data.status ?? null
      rollouts.value = data.rollouts ?? []
      if (selected.value) {
        const stillExists = rollouts.value.some((item) => item.rollout_id === selected.value?.rollout.rollout_id)
        if (stillExists) await selectRollout(selected.value.rollout.rollout_id)
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function selectRollout(rolloutId: string) {
    detailLoading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/post-review/rollouts/${encodeURIComponent(rolloutId)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      selected.value = data as ReviewDetail
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      detailLoading.value = false
    }
  }

  function clearSelection() {
    selected.value = null
  }

  return {
    status,
    rollouts,
    selected,
    loading,
    detailLoading,
    error,
    refresh,
    selectRollout,
    clearSelection,
  }
})
