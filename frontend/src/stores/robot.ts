import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface RobotProfileSummary {
  robot_id: string
  display_name: string
  profile_path: string
  schema_version: string
  connection_type: string
  ros_domain_id?: number | null
  namespace?: string
  active: boolean
  active_color_stream?: string | null
  active_robot_state?: string | null
  color_topic?: string | null
  color_msg_type?: string | null
  state_topic?: string | null
  state_msg_type?: string | null
  gripper_source?: string | null
  gripper_field?: string | null
  read_only: boolean
  policy_drive: boolean
}

export interface RobotStreamStatus {
  stream_id: string
  role: string
  ros_topic: string
  required: boolean
  present: boolean
  age_sec?: number | null
  payload_type?: string | null
  payload_shape?: number[] | null
  sample_meta?: Record<string, unknown> | null
}

export interface CanonicalSignals {
  primary_image?: {
    present: boolean
    topic?: string | null
    shape?: number[] | null
    age_sec?: number | null
  }
  active_eef_pose?: {
    present: boolean
    topic?: string | null
    xyz?: number[] | null
    rxyz?: number[] | null
    age_sec?: number | null
  }
  gripper?: {
    present: boolean
    topic?: string | null
    value?: number | null
    state?: string | null
    source?: string | null
    close_rule?: { operator: string, value: number } | null
  }
  vsa_binding?: {
    camera_topic?: string | null
    state_topic?: string | null
  }
}

export interface RobotValidation {
  status: string
  robot_id?: string
  checked_at?: string
  canonical_signals?: CanonicalSignals
  streams: RobotStreamStatus[]
}

export interface RobotProfileDetail {
  profile: RobotProfileSummary
  payload: Record<string, unknown>
  validation: RobotValidation
}

export interface RobotOnboardingResult {
  status: string
  job_id: string
  robot_id: string
  generated_profile_path: string
  artifact_profile_path: string
  report_path: string
  events_path: string
  report: Record<string, unknown>
  events: Record<string, unknown>[]
  validation?: RobotValidation
}

export const useRobotStore = defineStore('robot', () => {
  const profiles = ref<RobotProfileSummary[]>([])
  const activeRobotId = ref<string | null>(null)
  const activeProfilePath = ref<string | null>(null)
  const selected = ref<RobotProfileDetail | null>(null)
  const validation = ref<RobotValidation | null>(null)
  const loading = ref(false)
  const detailLoading = ref(false)
  const activating = ref(false)
  const validating = ref(false)
  const onboardingRunning = ref(false)
  const onboardingResult = ref<RobotOnboardingResult | null>(null)
  const error = ref<string | null>(null)

  async function refresh() {
    loading.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/robots')
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      profiles.value = data.profiles ?? []
      activeRobotId.value = data.active_robot_id ?? null
      activeProfilePath.value = data.active_profile_path ?? null
      const preferred = selected.value?.profile.robot_id ?? activeRobotId.value ?? profiles.value[0]?.robot_id
      if (preferred) await select(preferred)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function select(robotId: string) {
    detailLoading.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/robots/${encodeURIComponent(robotId)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      selected.value = data as RobotProfileDetail
      validation.value = selected.value.validation ?? null
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      detailLoading.value = false
    }
  }

  async function activate(robotId: string) {
    activating.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/robots/${encodeURIComponent(robotId)}/activate`, { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      activeRobotId.value = data.profile?.robot_id ?? robotId
      validation.value = data.validation ?? null
      await refresh()
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      activating.value = false
    }
  }

  async function validate(robotId = selected.value?.profile.robot_id ?? activeRobotId.value) {
    if (!robotId) return null
    validating.value = true
    error.value = null
    try {
      const resp = await fetch(`/api/session/robots/${encodeURIComponent(robotId)}/validate`, { method: 'POST' })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      validation.value = data as RobotValidation
      return data as RobotValidation
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      validating.value = false
    }
  }

  async function startOnboarding(profileYaml: string, robotNote = '') {
    onboardingRunning.value = true
    error.value = null
    try {
      const resp = await fetch('/api/session/robots/onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_yaml: profileYaml, robot_note: robotNote }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
      onboardingResult.value = data as RobotOnboardingResult
      if (onboardingResult.value.validation) validation.value = onboardingResult.value.validation
      await refresh()
      if (onboardingResult.value.robot_id) await select(onboardingResult.value.robot_id)
      return onboardingResult.value
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      onboardingRunning.value = false
    }
  }

  return {
    profiles,
    activeRobotId,
    activeProfilePath,
    selected,
    validation,
    loading,
    detailLoading,
    activating,
    validating,
    onboardingRunning,
    onboardingResult,
    error,
    refresh,
    select,
    activate,
    validate,
    startOnboarding,
  }
})
