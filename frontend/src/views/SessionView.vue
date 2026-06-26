<script setup lang="ts">
import { onMounted } from 'vue'
import { NCard, NButton, NSpace, NCode, NAlert } from 'naive-ui'
import { useSessionStore } from '@/stores/session'

const store = useSessionStore()
onMounted(() => store.refresh())
</script>

<template>
  <div class="space-y-4">
    <h2 class="text-2xl font-semibold">Session</h2>
    <NSpace>
      <NButton :loading="store.loading" type="primary" @click="store.refresh()">
        Refresh State
      </NButton>
    </NSpace>
    <NAlert v-if="store.lastError" type="error" :title="store.lastError" />
    <NCard title="Current Session">
      <NCode :code="JSON.stringify(store.session, null, 2)" language="json" />
    </NCard>
    <NCard title="MJPEG Preview">
      <img :src="'/mjpeg'" alt="AR stream" class="max-w-full rounded border" />
    </NCard>
  </div>
</template>
