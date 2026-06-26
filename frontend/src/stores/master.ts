import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface MasterStatus {
  available: boolean
  task_root?: string
  master_dir?: string
  last_review_trigger?: string | null
  last_review_at?: string | null
  last_error?: string | null
  state?: Record<string, unknown> | null
  review?: Record<string, unknown> | null
  understanding?: Record<string, unknown> | null
  memory?: Record<string, unknown>[]
  events?: Record<string, unknown>[]
  report?: string | null
  understanding_report?: string | null
  health_summary?: Record<string, unknown> | null
  paths?: Record<string, string>
}

export const useMasterStore = defineStore('master', () => {
  const status = ref<MasterStatus | null>(null)
  const lastError = ref<string | null>(null)
  const loading = ref(false)
  const reviewing = ref(false)

  async function refresh() {
    loading.value = true
    lastError.value = null
    try {
      const resp = await fetch('/api/session/master/status')
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      status.value = data as MasterStatus
    } catch (e) {
      lastError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function runReview() {
    reviewing.value = true
    lastError.value = null
    try {
      const resp = await fetch('/api/session/master/review', { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      status.value = data as MasterStatus
      return status.value
    } catch (e) {
      lastError.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      reviewing.value = false
    }
  }

  return { status, lastError, loading, reviewing, refresh, runReview }
})
