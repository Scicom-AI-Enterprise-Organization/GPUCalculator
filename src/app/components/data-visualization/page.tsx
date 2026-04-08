"use client";

import { Badge } from "@/components/ui/badge";
import { DollarSign, Users, TrendingUp, BarChart3, Activity, Gauge } from "lucide-react";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";

const dataVizSidebarSections = [
  {
    items: [
      { label: "Charts & Graphs", href: "#charts-graphs", icon: BarChart3 },
      { label: "Stats Cards", href: "#stats-cards", icon: TrendingUp },
      { label: "Progress Indicators", href: "#progress-indicators", icon: Activity },
      { label: "KPI Widgets", href: "#kpi-widgets", icon: Gauge },
    ],
  },
];

export default function DataVisualizationPage() {
  return (
    <TwoColumnLayout
      title="Data Visualization"
      sidebarSections={dataVizSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Components for displaying data, metrics, and analytics. Includes charts,
          stats cards, progress indicators, and KPI widgets.
        </p>
      </div>

      {/* Charts & Graphs */}
      <section id="charts-graphs" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Charts & Graphs</h2>
        <p className="mb-6 text-muted-foreground">
          Various chart types for visualizing data trends and comparisons.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Bar Chart" count={1}>
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Revenue</span>
                <Badge variant="secondary" className="h-5 text-xs">+12%</Badge>
              </div>
              <div className="flex h-16 items-end gap-1">
                {[40, 65, 45, 80, 55, 90, 70].map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t bg-primary/80"
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Line Chart" count={1}>
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Visitors</span>
                <Badge variant="secondary" className="h-5 text-xs">+8%</Badge>
              </div>
              <svg className="h-16 w-full" viewBox="0 0 100 40">
                <polyline
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-primary"
                  points="0,35 15,28 30,32 45,20 60,24 75,12 100,8"
                />
              </svg>
            </div>
          </ComponentCard>

          <ComponentCard title="Donut Chart" count={1}>
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card p-3">
              <div className="flex items-center justify-center">
                <div className="relative h-16 w-16">
                  <svg className="h-full w-full -rotate-90" viewBox="0 0 36 36">
                    <circle
                      cx="18"
                      cy="18"
                      r="14"
                      fill="none"
                      className="stroke-border"
                      strokeWidth="4"
                    />
                    <circle
                      cx="18"
                      cy="18"
                      r="14"
                      fill="none"
                      className="stroke-primary"
                      strokeWidth="4"
                      strokeDasharray="65 100"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                    65%
                  </div>
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Stats Cards */}
      <section id="stats-cards" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Stats Cards</h2>
        <p className="mb-6 text-muted-foreground">
          Compact cards for displaying key metrics and statistics.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Stats" count={1}>
            <div className="flex gap-2">
              <div className="rounded-lg border border-border bg-card p-3">
                <DollarSign className="mb-1 h-4 w-4 text-primary" />
                <div className="text-lg font-bold">$45.2k</div>
                <div className="text-xs text-muted-foreground">Revenue</div>
              </div>
              <div className="rounded-lg border border-border bg-card p-3">
                <Users className="mb-1 h-4 w-4 text-success-500" />
                <div className="text-lg font-bold">2,340</div>
                <div className="text-xs text-muted-foreground">Users</div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="With Trend" count={1}>
            <div className="rounded-lg border border-border bg-card p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Total Sales</span>
                <Badge className="bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-400 h-5 text-xs">
                  +12.5%
                </Badge>
              </div>
              <div className="mt-1 text-2xl font-bold">$128,430</div>
              <div className="mt-1 text-xs text-muted-foreground">vs last month</div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Progress Indicators */}
      <section id="progress-indicators" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Progress Indicators</h2>
        <p className="mb-6 text-muted-foreground">
          Visual indicators showing progress toward goals or completion status.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Linear Progress" count={1}>
            <div className="w-full max-w-[160px] space-y-3">
              <div>
                <div className="mb-1 flex justify-between text-xs">
                  <span>Sales</span>
                  <span>78%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div className="h-2 w-[78%] rounded-full bg-primary" />
                </div>
              </div>
              <div>
                <div className="mb-1 flex justify-between text-xs">
                  <span>Growth</span>
                  <span>45%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div className="h-2 w-[45%] rounded-full bg-success-500" />
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Circular Progress" count={1}>
            <div className="flex gap-4">
              <div className="relative h-14 w-14">
                <svg className="h-full w-full -rotate-90" viewBox="0 0 36 36">
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    className="stroke-border"
                    strokeWidth="3"
                  />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    className="stroke-primary"
                    strokeWidth="3"
                    strokeDasharray="78 100"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                  78%
                </div>
              </div>
              <div className="relative h-14 w-14">
                <svg className="h-full w-full -rotate-90" viewBox="0 0 36 36">
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    className="stroke-border"
                    strokeWidth="3"
                  />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    className="stroke-success-500"
                    strokeWidth="3"
                    strokeDasharray="45 100"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                  45%
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* KPI Widgets */}
      <section id="kpi-widgets" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">KPI Widgets</h2>
        <p className="mb-6 text-muted-foreground">
          Key Performance Indicator widgets with visual highlights and comparisons.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic KPI" count={1}>
            <div className="rounded-lg border border-border bg-card p-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-muted-foreground">Conversion</div>
                  <div className="text-xl font-bold">3.24%</div>
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-success-100 dark:bg-success-900/30">
                  <TrendingUp className="h-5 w-5 text-success-600" />
                </div>
              </div>
              <div className="mt-2 text-xs text-success-600">+0.5% from last week</div>
            </div>
          </ComponentCard>

          <ComponentCard title="KPI with Sparkline" count={1}>
            <div className="rounded-lg border border-border bg-card p-3">
              <div className="text-xs text-muted-foreground">Active Users</div>
              <div className="flex items-end justify-between">
                <div className="text-xl font-bold">12,543</div>
                <svg className="h-8 w-20" viewBox="0 0 80 32">
                  <polyline
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="text-primary"
                    points="0,28 10,24 20,26 30,18 40,20 50,12 60,16 70,8 80,4"
                  />
                </svg>
              </div>
              <div className="mt-1 text-xs text-success-600">+2.3% vs yesterday</div>
            </div>
          </ComponentCard>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
