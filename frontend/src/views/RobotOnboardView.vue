<script setup lang="ts">
import { computed, h, onMounted, reactive } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NCode,
  NDataTable,
  NDescriptions,
  NDescriptionsItem,
  NGrid,
  NGridItem,
  NInput,
  NTag,
  type DataTableColumns,
} from 'naive-ui'
import { useRobotStore, type RobotProfileSummary, type RobotStreamStatus } from '@/stores/robot'

const robot = useRobotStore()

const selectedId = computed(() => robot.selected?.profile.robot_id ?? robot.activeRobotId)
const selectedProfile = computed(() => robot.selected?.profile ?? null)
const validationStatus = computed(() => robot.validation?.status ?? 'not_checked')
const canonical = computed(() => robot.validation?.canonical_signals ?? null)
const onboardingForm = reactive({ profileYaml: '', robotNote: '' })
const onboardingReport = computed(() => robot.onboardingResult?.report ?? null)
const onboardingEvents = computed(() => robot.onboardingResult?.events ?? [])

const profileColumns: DataTableColumns<RobotProfileSummary> = [
  {
    title: 'Robot',
    key: 'display_name',
    render(row) {
      return h('div', { class: 'space-y-1' }, [
        h('div', { class: 'font-medium' }, row.display_name),
        h('div', { class: 'text-xs text-gray-400' }, row.robot_id),
      ])
    },
  },
  {
    title: 'Connection',
    key: 'connection_type',
    width: 130,
    render(row) {
      return h('span', {}, row.ros_domain_id == null ? row.connection_type : `${row.connection_type} / ${row.ros_domain_id}`)
    },
  },
  {
    title: 'State',
    key: 'active',
    width: 90,
    render(row) {
      return h(NTag, { size: 'small', type: row.active ? 'success' : 'default' }, () => row.active ? 'Active' : 'Ready')
    },
  },
]

const streamColumns: DataTableColumns<RobotStreamStatus> = [
  { title: 'Stream', key: 'stream_id', width: 150 },
  {
    title: 'Role',
    key: 'role',
    width: 130,
    render(row) {
      return h(NTag, { size: 'small', type: row.role === 'color_image' ? 'info' : 'warning' }, () => row.role)
    },
  },
  { title: 'ROS topic', key: 'ros_topic' },
  {
    title: 'Signal',
    key: 'present',
    width: 130,
    render(row) {
      return h('div', { class: 'flex items-center gap-2' }, [
        h(NTag, { size: 'small', type: row.present ? 'success' : 'error' }, () => row.present ? 'Live' : 'Waiting'),
        row.age_sec == null ? null : h('span', { class: 'text-xs text-gray-500' }, `${row.age_sec}s`),
      ])
    },
  },
]

function rowProps(row: RobotProfileSummary) {
  return {
    class: row.robot_id === selectedId.value ? 'cursor-pointer bg-blue-50' : 'cursor-pointer',
    onClick: () => robot.select(row.robot_id),
  }
}

function signalTag(present?: boolean) {
  return present ? 'success' : 'error'
}

function signalText(present?: boolean) {
  return present ? 'Live' : 'Waiting'
}

function fmtVec(value?: number[] | null) {
  return value?.length ? value.map((item) => Number(item).toFixed(4)).join(', ') : '-'
}

function fmtShape(value?: number[] | null) {
  return value?.length ? value.join(' x ') : '-'
}

function fmtRule(rule?: { operator: string, value: number } | null) {
  return rule ? `${rule.operator} ${rule.value}` : '-'
}

function displayValue(value: unknown) {
  if (value === null || value === undefined || value === '') return '-'
  return String(value)
}

function asJson(value: unknown) {
  return JSON.stringify(value ?? null, null, 2)
}

async function startAgentOnboarding() {
  await robot.startOnboarding(onboardingForm.profileYaml, onboardingForm.robotNote)
}

onMounted(() => {
  void robot.refresh()
})
</script>

<template>
  <div class="space-y-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 class="text-2xl font-semibold">Robot Onboard</h1>
        <p class="mt-1 text-sm text-gray-500">Select a robot profile and inspect live ROS2 source streams.</p>
      </div>
      <div class="flex gap-2">
        <NButton :loading="robot.loading" @click="robot.refresh()">Refresh</NButton>
        <NButton
          type="primary"
          :disabled="!selectedProfile"
          :loading="robot.activating"
          @click="selectedProfile && robot.activate(selectedProfile.robot_id)"
        >
          Activate Profile
        </NButton>
      </div>
    </div>

    <NAlert v-if="robot.error" type="error" closable>{{ robot.error }}</NAlert>

    <NCard title="Agent Onboarding" :bordered="false">
      <NGrid cols="1 l:5" :x-gap="16" :y-gap="16" responsive="screen">
        <NGridItem span="1 l:3">
          <div class="space-y-3">
            <NInput
              v-model:value="onboardingForm.profileYaml"
              type="textarea"
              placeholder="Paste RoboLineage.robot_profile.v1 YAML here"
              :autosize="{ minRows: 8, maxRows: 18 }"
            />
            <NInput
              v-model:value="onboardingForm.robotNote"
              placeholder="Optional robot note"
            />
            <div class="flex flex-wrap items-center gap-2">
              <NButton
                type="primary"
                :disabled="!onboardingForm.profileYaml.trim()"
                :loading="robot.onboardingRunning"
                @click="startAgentOnboarding"
              >
                Start Onboarding
              </NButton>
              <span class="text-xs text-gray-500">Generated profiles are written to robot_profiles/ and can be activated with the existing button.</span>
            </div>
          </div>
        </NGridItem>
        <NGridItem span="1 l:2">
          <div v-if="robot.onboardingResult" class="space-y-3">
            <NDescriptions bordered label-placement="left" :column="1" size="small">
              <NDescriptionsItem label="Status">
                <NTag size="small" :type="robot.onboardingResult.status === 'generated' ? 'success' : 'warning'">
                  {{ robot.onboardingResult.status }}
                </NTag>
              </NDescriptionsItem>
              <NDescriptionsItem label="Robot">{{ robot.onboardingResult.robot_id }}</NDescriptionsItem>
              <NDescriptionsItem label="Validation">
                {{ robot.onboardingResult.validation?.status ?? '-' }}
              </NDescriptionsItem>
              <NDescriptionsItem label="Active camera">
                {{ displayValue(onboardingReport?.active_camera) }}
              </NDescriptionsItem>
              <NDescriptionsItem label="Active arm">
                {{ displayValue(onboardingReport?.active_robot_state) }}
              </NDescriptionsItem>
              <NDescriptionsItem label="Generated profile">
                <span class="break-all text-xs">{{ robot.onboardingResult.generated_profile_path }}</span>
              </NDescriptionsItem>
            </NDescriptions>
            <NCode :code="asJson(onboardingEvents)" language="json" />
          </div>
          <div v-else class="text-sm text-gray-500">Paste a robot profile YAML to generate a normalized profile and validation report.</div>
        </NGridItem>
      </NGrid>
    </NCard>

    <NGrid cols="1 l:5" :x-gap="16" :y-gap="16" responsive="screen">
      <NGridItem span="1 l:2">
        <NCard title="Profiles" :bordered="false">
          <NDataTable
            :columns="profileColumns"
            :data="robot.profiles"
            :loading="robot.loading"
            :row-props="rowProps"
            :pagination="{ pageSize: 8 }"
          />
        </NCard>
      </NGridItem>

      <NGridItem span="1 l:3">
        <NCard title="Live Preview" :bordered="false">
          <div class="overflow-hidden rounded border border-gray-200 bg-black">
            <img :src="'/mjpeg'" class="aspect-video w-full object-contain" alt="live robot camera preview" />
          </div>
          <div class="mt-3 flex flex-wrap items-center justify-between gap-2">
            <div class="flex items-center gap-2 text-sm text-gray-600">
              <span>ROS2 source check</span>
              <NTag :type="validationStatus === 'ok' ? 'success' : 'warning'" size="small">
                {{ validationStatus }}
              </NTag>
            </div>
            <NButton
              :disabled="!selectedProfile"
              :loading="robot.validating"
              @click="selectedProfile && robot.validate(selectedProfile.robot_id)"
            >
              Check Streams
            </NButton>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <NCard v-if="selectedProfile" title="Canonical Signals" :bordered="false">
      <div class="mb-3 flex flex-wrap items-center gap-2 text-xs text-gray-500">
        <NTag size="small" type="info">{{ selectedProfile.robot_id }}</NTag>
        <span>{{ selectedProfile.profile_path }}</span>
      </div>
      <NDescriptions bordered label-placement="left" :column="2" size="small">
        <NDescriptionsItem label="Primary image">
          <div class="flex flex-wrap items-center gap-2">
            <NTag size="small" :type="signalTag(canonical?.primary_image?.present)">
              {{ signalText(canonical?.primary_image?.present) }}
            </NTag>
            <span>{{ canonical?.primary_image?.topic ?? '-' }}</span>
            <span class="text-gray-400">{{ canonical?.primary_image?.age_sec ?? '-' }}s</span>
          </div>
        </NDescriptionsItem>
        <NDescriptionsItem label="Image shape">{{ fmtShape(canonical?.primary_image?.shape) }}</NDescriptionsItem>
        <NDescriptionsItem label="Active EEF xyz">
          <div class="flex flex-wrap items-center gap-2">
            <NTag size="small" :type="signalTag(canonical?.active_eef_pose?.present)">
              {{ signalText(canonical?.active_eef_pose?.present) }}
            </NTag>
            <span>{{ fmtVec(canonical?.active_eef_pose?.xyz) }}</span>
          </div>
        </NDescriptionsItem>
        <NDescriptionsItem label="Active EEF rxyz">{{ fmtVec(canonical?.active_eef_pose?.rxyz) }}</NDescriptionsItem>
        <NDescriptionsItem label="Gripper">
          <div class="flex flex-wrap items-center gap-2">
            <NTag size="small" :type="signalTag(canonical?.gripper?.present)">
              {{ signalText(canonical?.gripper?.present) }}
            </NTag>
            <NTag size="small" :type="canonical?.gripper?.state === 'closed' ? 'warning' : 'success'">
              {{ canonical?.gripper?.state ?? 'unknown' }}
            </NTag>
            <span>{{ canonical?.gripper?.value ?? '-' }}</span>
          </div>
        </NDescriptionsItem>
        <NDescriptionsItem label="Gripper source">{{ canonical?.gripper?.source ?? selectedProfile.gripper_source ?? '-' }}</NDescriptionsItem>
        <NDescriptionsItem label="Close rule">{{ fmtRule(canonical?.gripper?.close_rule) }}</NDescriptionsItem>
        <NDescriptionsItem label="VSA binding">
          {{ canonical?.vsa_binding?.camera_topic ?? '-' }} / {{ canonical?.vsa_binding?.state_topic ?? '-' }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Connection">{{ selectedProfile.connection_type }} / {{ selectedProfile.ros_domain_id ?? '-' }}</NDescriptionsItem>
        <NDescriptionsItem label="Mode">
          <NTag size="small" :type="selectedProfile.read_only ? 'info' : 'warning'">
            {{ selectedProfile.read_only ? 'read only' : 'policy drive enabled' }}
          </NTag>
        </NDescriptionsItem>
      </NDescriptions>
    </NCard>

    <NCard title="ROS2 Source Streams" :bordered="false">
      <NDataTable
        :columns="streamColumns"
        :data="robot.validation?.streams ?? []"
        :loading="robot.detailLoading || robot.validating"
        :pagination="false"
      />
    </NCard>
  </div>
</template>
