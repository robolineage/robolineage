<script setup lang="ts">
import { computed, h } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { NLayout, NLayoutHeader, NLayoutSider, NLayoutContent, NMenu, type MenuOption } from 'naive-ui'

const route = useRoute()
const activeKey = computed(() => (route.name as string) ?? 'home')

const menuOptions: MenuOption[] = [
  {
    label: () => h(RouterLink, { to: '/robot-onboard' }, () => 'Robot Onboard'),
    key: 'robot-onboard',
  },
  {
    label: () => h(RouterLink, { to: '/rollout-control' }, () => 'Rollout Control'),
    key: 'rollout-control',
  },
  {
    label: () => h(RouterLink, { to: '/review-lifecycle' }, () => 'Review / Lifecycle'),
    key: 'review-lifecycle',
  },
  {
    label: () => h(RouterLink, { to: '/master' }, () => 'Master'),
    key: 'master',
  },
  {
    label: () => h(RouterLink, { to: '/health' }, () => 'Health'),
    key: 'health',
  },
]
</script>

<template>
  <NLayout class="h-full">
    <NLayoutHeader bordered class="flex h-14 items-center px-6">
      <span class="text-lg font-semibold">RoboLineage Console</span>
      <span class="ml-3 text-xs text-gray-400">Robot Policy Data Lifecycle Governance</span>
    </NLayoutHeader>
    <NLayout has-sider class="h-[calc(100%-3.5rem)]">
      <NLayoutSider
        bordered
        :width="200"
        :collapsed-width="64"
        show-trigger
        collapse-mode="width"
      >
        <NMenu :value="activeKey" :options="menuOptions" :indent="18" />
      </NLayoutSider>
      <NLayoutContent class="p-6">
        <RouterView />
      </NLayoutContent>
    </NLayout>
  </NLayout>
</template>
