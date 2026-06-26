<script setup lang="ts">
import { onMounted } from 'vue'
import { NCard, NButton, NSpace, NCode, NAlert } from 'naive-ui'
import { useHealthStore } from '@/stores/health'

const store = useHealthStore()
onMounted(() => store.refresh())
</script>

<template>
  <div class="space-y-4">
    <h2 class="text-2xl font-semibold">Health Check</h2>
    <NSpace>
      <NButton :loading="store.loading" type="primary" @click="store.refresh()">
        Refresh
      </NButton>
    </NSpace>
    <NAlert v-if="store.lastError" type="error" :title="store.lastError" />
    <NCard title="/health">
      <NCode :code="JSON.stringify(store.status, null, 2)" language="json" />
    </NCard>
  </div>
</template>
