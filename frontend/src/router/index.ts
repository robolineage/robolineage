import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('@/layouts/MainLayout.vue'),
    children: [
      {
        path: '',
        redirect: '/robot-onboard',
      },
      {
        path: 'robot-onboard',
        name: 'robot-onboard',
        component: () => import('@/views/RobotOnboardView.vue'),
        meta: { title: 'Robot Onboard' },
      },
      {
        path: 'rollout-control',
        name: 'rollout-control',
        component: () => import('@/views/HomeView.vue'),
        meta: { title: 'Rollout Control' },
      },
      {
        path: 'collection',
        redirect: '/rollout-control',
      },
      {
        path: 'dashboard',
        name: 'dashboard',
        component: () => import('@/views/DashboardView.vue'),
        meta: { title: 'Overview' },
      },
      {
        path: 'session',
        name: 'session',
        component: () => import('@/views/SessionView.vue'),
        meta: { title: 'Session' },
      },
      {
        path: 'review-lifecycle',
        name: 'review-lifecycle',
        component: () => import('@/views/ReviewLifecycleView.vue'),
        meta: { title: 'Review / Lifecycle' },
      },
      {
        path: 'master',
        name: 'master',
        component: () => import('@/views/MasterView.vue'),
        meta: { title: 'Master' },
      },
      {
        path: 'health',
        name: 'health',
        component: () => import('@/views/HealthView.vue'),
        meta: { title: 'Health' },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
  },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
