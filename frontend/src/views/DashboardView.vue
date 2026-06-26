<script setup lang="ts">
import { onMounted } from 'vue'
import { NCard, NGrid, NGridItem, NStatistic, NTag } from 'naive-ui'
import { useSessionStore } from '@/stores/session'
import { useHealthStore } from '@/stores/health'

const sessionStore = useSessionStore()
const healthStore = useHealthStore()

onMounted(() => {
  sessionStore.refresh()
  healthStore.refresh()
})
</script>

<template>
  <div class="space-y-4">
    <h2 class="text-2xl font-semibold">Overview</h2>
    <NGrid :cols="3" :x-gap="16" :y-gap="16">
      <NGridItem>
        <NCard title="Session State">
          <NStatistic :value="sessionStore.session.state">
            <template #suffix>
              <NTag :type="sessionStore.session.state === 'IDLE' ? 'default' : 'success'" size="small">
                {{ sessionStore.session.mode ?? '—' }}
              </NTag>
            </template>
          </NStatistic>
          <p class="mt-2 text-xs text-gray-500">
            rollout_id: {{ sessionStore.session.rollout_id ?? '—' }}
          </p>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard title="Health Check">
          <NTag :type="healthStore.lastError ? 'error' : 'success'">
            {{ healthStore.lastError ? 'Unavailable' : 'OK' }}
          </NTag>
          <p class="mt-2 text-xs text-gray-500 break-all">
            {{ healthStore.lastError ?? JSON.stringify(healthStore.status) }}
          </p>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard title="Backend Entry">
          <ul class="text-sm space-y-1">
            <li><code>/api/health</code> → :8081/health</li>
            <li><code>/api/session/**</code> → :8080/**</li>
            <li><code>/stream</code> → :8080</li>
            <li><code>/mjpeg</code> → :8080</li>
          </ul>
        </NCard>
      </NGridItem>
    </NGrid>
  </div>
</template>
