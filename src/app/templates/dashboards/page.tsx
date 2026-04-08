"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TwoColumnLayout } from "@/components/two-column-layout";
import {
  ArrowLeft,
  ArrowRight,
  Eye,
  LayoutDashboard,
  BarChart3,
  Users,
  ShoppingCart,
  DollarSign,
  Activity,
  Bell,
  Search,
  Settings,
  TrendingUp,
  PieChart,
} from "lucide-react";

const dashboardSidebarSections = [
  {
    items: [
      { label: "Analytics Dashboard", href: "#analytics", icon: BarChart3 },
      { label: "Admin Dashboard", href: "#admin", icon: LayoutDashboard },
      { label: "Team Dashboard", href: "#team", icon: Users },
      { label: "Sales Dashboard", href: "#sales", icon: DollarSign },
      { label: "Overview Dashboard", href: "#overview", icon: PieChart },
    ],
  },
];

const dashboardTemplates = [
  {
    id: "analytics",
    title: "Analytics Dashboard",
    description:
      "Comprehensive analytics with interactive charts, KPIs, and real-time metrics. Perfect for business intelligence and data-driven decision making.",
    icon: BarChart3,
    features: [
      "Interactive charts",
      "Real-time data feeds",
      "Export to CSV/PDF",
      "Custom KPI cards",
      "Date range filtering",
    ],
    status: "Popular",
  },
  {
    id: "admin",
    title: "Admin Dashboard",
    description:
      "Full-featured admin panel with user management, system settings, and monitoring. Built for managing complex applications.",
    icon: LayoutDashboard,
    features: [
      "User management table",
      "System health monitor",
      "Activity logs",
      "Role-based access",
      "Settings panel",
    ],
    status: "New",
  },
  {
    id: "team",
    title: "Team Dashboard",
    description:
      "Team collaboration workspace with task tracking, member activity, and project overview for agile teams.",
    icon: Users,
    features: [
      "Kanban task board",
      "Team member cards",
      "Project timeline",
      "Resource allocation",
      "Sprint metrics",
    ],
    status: null,
  },
  {
    id: "sales",
    title: "Sales Dashboard",
    description:
      "Revenue tracking and sales pipeline management with forecasting and conversion analytics.",
    icon: DollarSign,
    features: [
      "Revenue charts",
      "Pipeline view",
      "Conversion funnels",
      "Sales leaderboard",
      "Forecast models",
    ],
    status: null,
  },
  {
    id: "overview",
    title: "Overview Dashboard",
    description:
      "High-level executive dashboard with key metrics, trend summaries, and department snapshots.",
    icon: PieChart,
    features: [
      "KPI summary cards",
      "Trend indicators",
      "Department breakdown",
      "Goal tracking",
      "Notification center",
    ],
    status: "New",
  },
];

/* ---------- Chart Data ---------- */
const chartDataSets: Record<string, number[]> = {
  "7D": [55, 70, 45, 85, 60, 75, 90],
  "1M": [40, 65, 45, 80, 55, 90, 70, 85, 60, 75, 95, 68],
  "3M": [
    30, 45, 60, 50, 70, 55, 80, 65, 75, 90, 85, 70, 60, 80, 95, 72, 88, 65,
    78, 55, 90, 82, 68, 75,
  ],
};

/* ---------- Interactive Dashboard Preview ---------- */
function DashboardPreviewDemo() {
  const [activeSidebarItem, setActiveSidebarItem] = useState(0);
  const [chartPeriod, setChartPeriod] = useState<"7D" | "1M" | "3M">("1M");

  const sidebarIcons = [
    LayoutDashboard,
    BarChart3,
    Users,
    ShoppingCart,
    Activity,
    Bell,
    Settings,
  ];
  const statCards = [
    {
      icon: DollarSign,
      label: "Revenue",
      value: "$45.2k",
      change: "+12.5%",
    },
    { icon: Users, label: "Users", value: "2,340", change: "+8.1%" },
    {
      icon: ShoppingCart,
      label: "Orders",
      value: "1,210",
      change: "+3.2%",
    },
    {
      icon: TrendingUp,
      label: "Growth",
      value: "18.2%",
      change: "+2.4%",
    },
  ];
  const chartData = chartDataSets[chartPeriod];
  const activityItems = [
    { name: "Sarah Chen", action: "New order", time: "2m ago" },
    { name: "Alex Kim", action: "Payment received", time: "5m ago" },
    { name: "Jordan Lee", action: "User registered", time: "12m ago" },
    { name: "Sam Wilson", action: "Item shipped", time: "18m ago" },
    { name: "Maya Patel", action: "Review submitted", time: "25m ago" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mx-auto mt-10 max-w-4xl overflow-hidden rounded-xl border border-border shadow-2xl"
    >
      {/* Browser Chrome */}
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          scicom-design-hub.vercel.app/dashboard
        </div>
      </div>

      <div className="flex h-[560px] bg-background">
        {/* Sidebar */}
        <div className="flex w-14 shrink-0 flex-col items-center gap-3 border-r border-border bg-card py-4">
          <div className="mb-2 h-8 w-8 rounded-lg bg-primary/20" />
          {sidebarIcons.map((Icon, i) => (
            <motion.div
              key={i}
              whileHover={{ scale: 1.15 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setActiveSidebarItem(i)}
              className={`flex h-9 w-9 cursor-pointer items-center justify-center rounded-lg transition-colors ${
                activeSidebarItem === i
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <Icon className="h-4 w-4" />
            </motion.div>
          ))}
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {/* Header */}
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-foreground">
                Dashboard
              </h3>
              <p className="text-xs text-muted-foreground">
                Welcome back, Sarah
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground">
                <Search className="h-3 w-3" />
                Search...
              </div>
              <div className="relative">
                <Bell className="h-4 w-4 text-muted-foreground" />
                <div className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-red-500" />
              </div>
              <div className="h-7 w-7 rounded-full bg-primary/30" />
            </div>
          </div>

          {/* Stat Cards */}
          <div className="mb-5 grid grid-cols-4 gap-3">
            {statCards.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className="cursor-pointer rounded-lg border border-border bg-card p-3"
              >
                <div className="mb-1 flex items-center justify-between">
                  <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-[10px] text-green-500">
                    {stat.change}
                  </span>
                </div>
                <div className="text-xl font-bold text-foreground">
                  {stat.value}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>

          {/* Chart + Activity */}
          <div className="grid grid-cols-5 gap-4">
            {/* Chart */}
            <div className="col-span-3 rounded-lg border border-border bg-card p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">
                  Revenue Overview
                </span>
                <div className="flex gap-1">
                  {(["7D", "1M", "3M"] as const).map((period) => (
                    <div
                      key={period}
                      onClick={() => setChartPeriod(period)}
                      className={`cursor-pointer rounded-md px-2 py-1 text-xs transition-colors ${
                        chartPeriod === period
                          ? "bg-primary/10 font-medium text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {period}
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex h-40 items-end gap-1">
                {chartData.map((value, i) => (
                  <motion.div
                    key={`${chartPeriod}-${i}`}
                    initial={{ height: 0 }}
                    animate={{ height: `${value}%` }}
                    transition={{ delay: i * 0.03, duration: 0.4 }}
                    className="flex-1 rounded-t bg-primary/70 transition-colors hover:bg-primary"
                  />
                ))}
              </div>
            </div>

            {/* Activity */}
            <div className="col-span-2 rounded-lg border border-border bg-card p-4">
              <span className="mb-3 block text-sm font-medium text-foreground">
                Recent Activity
              </span>
              <div className="space-y-3">
                {activityItems.map((item, i) => (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="flex items-center gap-2"
                  >
                    <div className="h-6 w-6 shrink-0 rounded-full bg-muted" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium text-foreground">
                        {item.name}
                      </div>
                      <div className="truncate text-[10px] text-muted-foreground">
                        {item.action}
                      </div>
                    </div>
                    <span className="shrink-0 text-[10px] text-muted-foreground">
                      {item.time}
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom Section — Stats Table */}
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-border bg-card p-4">
              <span className="mb-2 block text-sm font-medium text-foreground">
                Top Products
              </span>
              <div className="space-y-2">
                {[
                  { name: "Enterprise Plan", revenue: "$12,400", pct: 85 },
                  { name: "Pro Plan", revenue: "$8,200", pct: 65 },
                  { name: "Starter Plan", revenue: "$4,100", pct: 40 },
                ].map((product) => (
                  <div key={product.name} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-foreground">{product.name}</span>
                      <span className="text-muted-foreground">
                        {product.revenue}
                      </span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-muted">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${product.pct}%` }}
                        transition={{ delay: 0.5, duration: 0.6 }}
                        className="h-full rounded-full bg-primary"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-4">
              <span className="mb-2 block text-sm font-medium text-foreground">
                Team Members
              </span>
              <div className="space-y-2">
                {[
                  { name: "Sarah Chen", role: "Designer", color: "bg-primary" },
                  { name: "Alex Kim", role: "Developer", color: "bg-primary/70" },
                  { name: "Jordan Lee", role: "Manager", color: "bg-primary/50" },
                ].map((member) => (
                  <motion.div
                    key={member.name}
                    whileHover={{ x: 4 }}
                    className="flex cursor-pointer items-center gap-2 rounded-md p-1 transition-colors hover:bg-muted"
                  >
                    <div
                      className={`flex h-7 w-7 items-center justify-center rounded-full ${member.color} text-[10px] font-medium text-primary-foreground`}
                    >
                      {member.name
                        .split(" ")
                        .map((n) => n[0])
                        .join("")}
                    </div>
                    <div>
                      <div className="text-xs font-medium text-foreground">
                        {member.name}
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        {member.role}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/* ---------- Template Card ---------- */
interface TemplateCardProps {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  features: string[];
  status: string | null;
}

function TemplateCard({
  title,
  description,
  icon: Icon,
  features,
  status,
}: TemplateCardProps) {
  return (
    <Card className="h-full transition-all hover:-translate-y-1 hover:shadow-lg">
      <CardHeader>
        <div className="mb-2 flex items-center justify-between">
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <Icon className="h-6 w-6 text-primary" />
          </div>
          {status && (
            <Badge variant={status === "New" ? "default" : "secondary"}>
              {status}
            </Badge>
          )}
        </div>
        <CardTitle className="text-xl">{title}</CardTitle>
        <CardDescription className="text-sm">{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ul className="space-y-2">
          {features.map((feature) => (
            <li
              key={feature}
              className="flex items-center gap-2 text-sm text-muted-foreground"
            >
              <div className="h-1.5 w-1.5 rounded-full bg-primary" />
              {feature}
            </li>
          ))}
        </ul>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="flex-1">
            <Eye className="mr-2 h-4 w-4" />
            Preview
          </Button>
          <Button size="sm" className="flex-1">
            See more
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function DashboardsPage() {
  return (
    <TwoColumnLayout
      title="Dashboards"
      sidebarSections={dashboardSidebarSections}
    >
      <div className="mb-8">
        <Link
          href="/templates"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Templates
        </Link>
      </div>

      <div className="mb-12">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <LayoutDashboard className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Dashboard Templates</h2>
            <p className="text-sm text-muted-foreground">
              Data visualization & admin panels
            </p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          Data-rich dashboard templates for analytics, admin panels, and team
          workspaces. Each template features interactive charts, real-time data
          feeds, and responsive layouts.
        </p>
      </div>

      {/* Interactive Preview */}
      <section className="mb-12">
        <h3 className="mb-2 text-xl font-semibold">Live Preview</h3>
        <p className="mb-4 text-sm text-muted-foreground">
          Interactive preview of the analytics dashboard. Click sidebar icons,
          switch chart periods, and hover over elements.
        </p>
        <DashboardPreviewDemo />
      </section>

      {/* Template Cards */}
      <section className="mb-0">
        <h3 className="mb-6 text-xl font-semibold">All Dashboard Templates</h3>
        <div className="grid gap-6 md:grid-cols-2">
          {dashboardTemplates.map((template) => (
            <div key={template.id} id={template.id}>
              <TemplateCard {...template} />
            </div>
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
