import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface SessionState {
  state: string
  rollout_id: string | null
  mode: string | null
}

export const useSessionStore = defineStore('session', () => {
  const session = ref<SessionState>({ state: 'IDLE', rollout_id: null, mode: null })
  const lastError = ref<string | null>(null)
  const loading = ref(false)

  async function refresh() {
    loading.value = true
    lastError.value = null
    try {
      const resp = await fetch('/api/session/state')
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      session.value = await resp.json()
    } catch (e) {
      lastError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  return { session, lastError, loading, refresh }
})
