<script setup lang="ts">
import { NResult, NButton } from 'naive-ui'
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<template>
  <NResult status="404" title="404" description="Page not found">
    <template #footer>
      <NButton @click="router.push('/')">Back Home</NButton>
    </template>
  </NResult>
</template>
