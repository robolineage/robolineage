import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useHealthStore = defineStore('health', () => {
  const status = ref<Record<string, unknown> | null>(null)
  const lastError = ref<string | null>(null)
  const loading = ref(false)

  async function refresh() {
    loading.value = true
    lastError.value = null
    try {
      const resp = await fetch('/api/health')
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      status.value = await resp.json()
    } catch (e) {
      lastError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  return { status, lastError, loading, refresh }
})
