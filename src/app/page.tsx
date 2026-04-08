"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import {
  ArrowRight,
  Palette,
  Component,
  BookOpen,
  BarChart3,
  PieChart,
  TrendingUp,
  Bell,
  Search,
  Settings,
  User,
  ChevronRight,
  LayoutDashboard,
  LogIn,
  Lock,
  Mail,
  Eye,
  Users,
  DollarSign,
  ShoppingCart,
  Activity,
} from "lucide-react";
import { motion } from "framer-motion";
import { PageHeader } from "@/components/page-header";
import { PageFooter } from "@/components/page-footer";
import {
  fadeIn,
  staggerContainer,
  defaultViewport,
} from "@/lib/animations";

const templateTabs = [
  { id: "solutions", label: "Solutions", icon: BookOpen },
  { id: "login", label: "Login", icon: LogIn },
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
] as const;

type TemplateTab = (typeof templateTabs)[number]["id"];

const solutionsPages: Record<string, { title: string; desc: string; stats: { label: string; value: string }[]; bars: number[] }> = {
  "AI Monitoring": { title: "AI Monitoring System", desc: "Real-time monitoring and analytics for your AI-powered contact center.", stats: [{ label: "Active Sessions", value: "1,247" }, { label: "Avg Response", value: "1.2s" }, { label: "Satisfaction", value: "94.8%" }], bars: [70, 85, 60, 90, 45, 80, 55, 95] },
  "Voice Agent": { title: "Voice Agent Console", desc: "Manage and configure voice-based AI agents for customer interactions.", stats: [{ label: "Active Calls", value: "342" }, { label: "Avg Duration", value: "4.5m" }, { label: "Resolution", value: "87.2%" }], bars: [50, 70, 80, 60, 90, 45, 75, 65] },
  "Agent Lobby": { title: "Agent Lobby", desc: "Real-time agent queue management and assignment dashboard.", stats: [{ label: "Agents Online", value: "48" }, { label: "In Queue", value: "12" }, { label: "Avg Wait", value: "23s" }], bars: [90, 60, 40, 75, 85, 50, 70, 80] },
  "Analytics": { title: "Analytics Dashboard", desc: "Comprehensive analytics and reporting across all channels.", stats: [{ label: "Total Interactions", value: "24.5k" }, { label: "CSAT Score", value: "4.7/5" }, { label: "First Contact", value: "76%" }], bars: [65, 80, 55, 70, 90, 85, 45, 75] },
};
const allSolutionItems = ["AI Monitoring", "Voice Agent", "Agent Lobby", "Analytics"];

function SolutionsPreview() {
  const [activePage, setActivePage] = useState("AI Monitoring");
  const page = solutionsPages[activePage];

  return (
    <div className="flex" style={{ height: 620 }}>
      <div className="w-64 shrink-0 border-r border-border bg-muted/30 p-5">
        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }} className="mb-6 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
            <BookOpen className="h-4 w-4 text-primary" />
          </div>
          <div>
            <p className="text-sm font-semibold">Solutions</p>
            <p className="text-xs text-muted-foreground">v1.0.1</p>
          </div>
        </motion.div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Contact Center</p>
        {allSolutionItems.map((item, i) => (
          <motion.div
            key={item}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2, delay: i * 0.05 }}
            onClick={() => setActivePage(item)}
            className={`mb-0.5 cursor-pointer rounded-md px-3 py-2 text-sm transition-colors ${activePage === item ? "bg-primary/10 font-medium text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}
          >
            {item}
          </motion.div>
        ))}
        <p className="mb-2 mt-5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Management</p>
        {["Knowledge Base", "Ticketing", "Blujay CRM"].map((item) => (
          <div key={item} className="mb-0.5 rounded-md px-3 py-2 text-sm text-muted-foreground">{item}</div>
        ))}
        <p className="mb-2 mt-5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Resources</p>
        {["Figma Tokens", "Documentation"].map((item) => (
          <div key={item} className="mb-0.5 rounded-md px-3 py-2 text-sm text-muted-foreground">{item}</div>
        ))}
      </div>
      <div className="flex-1 overflow-hidden p-6">
        <div className="mb-6 flex items-center gap-2 text-sm text-muted-foreground">
          <Component className="h-4 w-4" />
          <span>Contact Center</span>
          <ChevronRight className="h-3 w-3" />
          <span className="text-foreground">{activePage}</span>
        </div>
        <motion.div key={activePage} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <h3 className="mb-1 text-lg font-semibold">{page.title}</h3>
          <p className="mb-5 text-sm text-muted-foreground">{page.desc}</p>
          <div className="grid grid-cols-3 gap-4">
            {page.stats.map((stat, i) => (
              <motion.div key={stat.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: i * 0.1 }} className="rounded-xl border border-border bg-card p-4">
                <p className="text-xs text-muted-foreground">{stat.label}</p>
                <p className="mt-1 text-xl font-bold">{stat.value}</p>
              </motion.div>
            ))}
          </div>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.3 }} className="mt-4 rounded-xl border border-border bg-card p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-sm font-semibold">Session Timeline</p>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 animate-pulse rounded-full bg-green-500" />
                <span className="text-xs text-muted-foreground">Live</span>
              </div>
            </div>
            <div className="flex h-32 items-end gap-1.5 pt-2">
              {page.bars.map((h, i) => (
                <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${h}%` }} transition={{ duration: 0.5, delay: i * 0.06, ease: "easeOut" }} className="flex-1 rounded-t-md bg-primary/60" />
              ))}
            </div>
          </motion.div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.4 }} className="rounded-xl border border-border bg-card p-4">
              <p className="mb-2 text-sm font-semibold">Top Intents</p>
              <div className="space-y-2">
                {["Billing inquiry", "Technical support", "Account changes"].map((item, i) => (
                  <div key={item} className="flex items-center gap-2">
                    <div className="h-2 flex-1 rounded-full bg-muted">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${90 - i * 20}%` }} transition={{ duration: 0.6, delay: 0.5 + i * 0.1, ease: "easeOut" }} className="h-full rounded-full bg-primary/60" />
                    </div>
                    <span className="w-28 text-xs text-muted-foreground">{item}</span>
                  </div>
                ))}
              </div>
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.5 }} className="rounded-xl border border-border bg-card p-4">
              <p className="mb-2 text-sm font-semibold">Agent Status</p>
              <div className="space-y-2">
                {[
                  { name: "Online", count: 24, color: "bg-green-500" },
                  { name: "Busy", count: 12, color: "bg-yellow-500" },
                  { name: "Offline", count: 5, color: "bg-muted-foreground/40" },
                ].map((s) => (
                  <div key={s.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`h-2 w-2 rounded-full ${s.color}`} />
                      <span className="text-xs">{s.name}</span>
                    </div>
                    <span className="text-xs font-medium">{s.count}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function LoginPreview() {
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [rememberMe, setRememberMe] = useState(true);

  return (
    <div className="relative flex" style={{ height: 620 }}>
      {/* Left: Hero background image */}
      <div className="relative w-1/2 overflow-hidden">
        <Image src="/images/hero-bg-light.jpg" alt="" fill className="object-cover dark:hidden" />
        <Image src="/images/hero-bg.jpg" alt="" fill className="hidden object-cover dark:block" />
        <div className="absolute inset-0 bg-white/20 dark:bg-black/30" />
        <div className="relative z-10 flex h-full flex-col justify-end p-8">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.3 }} className="rounded-xl bg-background/70 p-5 backdrop-blur-md">
            <h3 className="mb-1 text-lg font-bold text-foreground">Scicom Design Hub</h3>
            <p className="text-sm text-muted-foreground">Build beautiful interfaces with our comprehensive design system and component library.</p>
          </motion.div>
        </div>
      </div>
      {/* Right: Login form */}
      <div className="flex flex-1 items-center justify-center bg-card p-8">
        <div className="w-full max-w-sm">
          <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.4, type: "spring" }} className="mb-8 flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
              <Lock className="h-6 w-6 text-primary" />
            </div>
          </motion.div>
          <motion.h3 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.1 }} className="mb-1 text-center text-xl font-semibold">Welcome back</motion.h3>
          <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.15 }} className="mb-8 text-center text-sm text-muted-foreground">Sign in to your account</motion.p>
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.2 }} className="space-y-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">Email</label>
              <div
                onClick={() => setFocusedField("email")}
                className={`flex cursor-text items-center gap-2 rounded-lg border bg-background px-3 py-2.5 transition-all ${focusedField === "email" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}
              >
                <Mail className={`h-4 w-4 transition-colors ${focusedField === "email" ? "text-primary" : "text-muted-foreground"}`} />
                <span className="text-sm text-muted-foreground">name@company.com</span>
              </div>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">Password</label>
              <div
                onClick={() => setFocusedField("password")}
                className={`flex cursor-text items-center gap-2 rounded-lg border bg-background px-3 py-2.5 transition-all ${focusedField === "password" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}
              >
                <Lock className={`h-4 w-4 transition-colors ${focusedField === "password" ? "text-primary" : "text-muted-foreground"}`} />
                <span className="flex-1 text-sm text-muted-foreground">••••••••</span>
                <Eye className="h-4 w-4 cursor-pointer text-muted-foreground transition-colors hover:text-foreground" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <label onClick={() => setRememberMe(!rememberMe)} className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
                <div className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${rememberMe ? "border-primary bg-primary" : "border-border bg-background"}`}>
                  {rememberMe && <svg className="h-3 w-3 text-primary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>}
                </div>
                Remember me
              </label>
              <span className="cursor-pointer text-sm text-primary transition-opacity hover:opacity-80">Forgot?</span>
            </div>
            <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} className="flex h-10 w-full cursor-pointer items-center justify-center rounded-lg bg-primary text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90">
              Sign in
            </motion.div>
            <div className="flex items-center gap-3">
              <div className="h-px flex-1 bg-border" />
              <span className="text-xs text-muted-foreground">or continue with</span>
              <div className="h-px flex-1 bg-border" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              {["Google", "Microsoft"].map((provider) => (
                <motion.div key={provider} whileHover={{ scale: 1.02, backgroundColor: "var(--muted)" }} whileTap={{ scale: 0.98 }} className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-border py-2.5 text-sm text-muted-foreground transition-colors">{provider}</motion.div>
              ))}
            </div>
            <p className="mt-4 text-center text-xs text-muted-foreground">
              Don&apos;t have an account? <span className="cursor-pointer text-primary transition-opacity hover:opacity-80">Sign up</span>
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

const chartDataSets: Record<string, number[]> = {
  "7D": [55, 70, 45, 85, 60, 75, 90],
  "1M": [40, 65, 45, 80, 55, 90, 70, 85, 60, 75, 95, 68],
  "3M": [30, 45, 60, 50, 70, 55, 80, 65, 75, 90, 85, 70, 60, 80, 95, 72, 88, 65, 78, 55, 90, 82, 68, 75],
};

function DashboardPreview() {
  const [activeSidebarItem, setActiveSidebarItem] = useState(0);
  const [chartPeriod, setChartPeriod] = useState("1M");
  const sidebarIcons = [LayoutDashboard, Users, ShoppingCart, BarChart3, Bell, Settings];
  const bars = chartDataSets[chartPeriod];

  return (
    <div className="flex" style={{ height: 620 }}>
      {/* Narrow sidebar */}
      <div className="w-16 shrink-0 border-r border-border bg-muted/30 p-2 pt-4">
        <div className="mb-6 flex justify-center">
          <div className="h-8 w-8 rounded-lg bg-primary/20" />
        </div>
        {sidebarIcons.map((Icon, i) => (
          <motion.div
            key={i}
            onClick={() => setActiveSidebarItem(i)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            className={`mb-1 flex h-10 w-full cursor-pointer items-center justify-center rounded-lg transition-colors ${activeSidebarItem === i ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}
          >
            <Icon className="h-4 w-4" />
          </motion.div>
        ))}
      </div>
      {/* Main content */}
      <div className="flex-1 overflow-hidden p-5">
        <motion.div initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="mb-5 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold">Dashboard</h3>
            <p className="text-xs text-muted-foreground">Welcome back, Sarah</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5">
              <Search className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Search...</span>
            </div>
            <div className="relative flex h-8 w-8 items-center justify-center rounded-full bg-muted">
              <Bell className="h-3.5 w-3.5 text-muted-foreground" />
              <div className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full border-2 border-card bg-destructive" />
            </div>
            <div className="h-8 w-8 rounded-full bg-primary/20" />
          </div>
        </motion.div>
        {/* Stat cards */}
        <div className="mb-5 grid grid-cols-4 gap-3">
          {[
            { label: "Revenue", value: "$45.2k", icon: DollarSign, change: "+12.5%" },
            { label: "Users", value: "2,340", icon: Users, change: "+8.1%" },
            { label: "Orders", value: "1,210", icon: ShoppingCart, change: "+3.2%" },
            { label: "Growth", value: "18.2%", icon: Activity, change: "+2.4%" },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: i * 0.08 }}
              whileHover={{ y: -2, transition: { duration: 0.15 } }}
              className="cursor-default rounded-xl border border-border bg-card p-3 transition-shadow hover:shadow-md"
            >
              <div className="mb-2 flex items-center justify-between">
                <stat.icon className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs text-green-500">{stat.change}</span>
              </div>
              <p className="text-lg font-bold">{stat.value}</p>
              <p className="text-xs text-muted-foreground">{stat.label}</p>
            </motion.div>
          ))}
        </div>
        {/* Chart + activity */}
        <div className="mb-4 grid grid-cols-5 gap-3">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.35 }} className="col-span-3 rounded-xl border border-border bg-card p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-sm font-semibold">Revenue Overview</p>
              <div className="flex gap-1">
                {["7D", "1M", "3M"].map((period) => (
                  <div
                    key={period}
                    onClick={() => setChartPeriod(period)}
                    className={`cursor-pointer rounded-md px-2 py-1 text-xs transition-colors ${chartPeriod === period ? "bg-primary/10 font-medium text-primary" : "text-muted-foreground hover:text-foreground"}`}
                  >
                    {period}
                  </div>
                ))}
              </div>
            </div>
            <div className="flex h-36 items-end gap-1 pt-2">
              {bars.map((h, i) => (
                <motion.div
                  key={`${chartPeriod}-${i}`}
                  initial={{ height: 0 }}
                  animate={{ height: `${h}%` }}
                  transition={{ duration: 0.4, delay: i * 0.03, ease: "easeOut" }}
                  className="group relative flex-1 cursor-pointer rounded-t-md bg-primary/20"
                >
                  <motion.div initial={{ height: 0 }} animate={{ height: "66%" }} transition={{ duration: 0.5, delay: 0.1 + i * 0.03, ease: "easeOut" }} className="w-full rounded-t-md bg-primary/60 transition-colors group-hover:bg-primary/80" />
                </motion.div>
              ))}
            </div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.4 }} className="col-span-2 rounded-xl border border-border bg-card p-4">
            <p className="mb-3 text-sm font-semibold">Recent Activity</p>
            <div className="space-y-3">
              {[
                { name: "Sarah Chen", action: "New order #1234", time: "2m ago" },
                { name: "Alex Kim", action: "Payment received", time: "5m ago" },
                { name: "Jordan Lee", action: "User registered", time: "12m ago" },
                { name: "Sam Wilson", action: "Item shipped", time: "18m ago" },
                { name: "Maya Patel", action: "Refund processed", time: "25m ago" },
              ].map((item, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.2, delay: 0.5 + i * 0.06 }} className="flex items-center gap-3 rounded-md p-1 transition-colors hover:bg-muted/50">
                  <div className="h-7 w-7 shrink-0 rounded-full bg-muted" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium">{item.name}</p>
                    <p className="truncate text-xs text-muted-foreground">{item.action}</p>
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">{item.time}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
        {/* Bottom row: table */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.5 }} className="rounded-xl border border-border bg-card p-4">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-semibold">Recent Orders</p>
            <span className="cursor-pointer text-xs text-primary transition-opacity hover:opacity-80">View all</span>
          </div>
          <div className="space-y-0">
            <div className="flex items-center gap-4 border-b border-border py-2 text-xs font-medium text-muted-foreground">
              <span className="w-20">Order</span>
              <span className="flex-1">Customer</span>
              <span className="w-20">Status</span>
              <span className="w-20 text-right">Amount</span>
            </div>
            {[
              { id: "#1234", customer: "Sarah Chen", status: "Completed", amount: "$240.00", color: "text-green-500" },
              { id: "#1233", customer: "Alex Kim", status: "Pending", amount: "$180.50", color: "text-yellow-500" },
              { id: "#1232", customer: "Jordan Lee", status: "Completed", amount: "$320.00", color: "text-green-500" },
            ].map((order, i) => (
              <motion.div key={order.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2, delay: 0.6 + i * 0.08 }} className="flex items-center gap-4 border-b border-border/50 py-2.5 text-xs transition-colors hover:bg-muted/30">
                <span className="w-20 font-medium">{order.id}</span>
                <span className="flex-1 text-muted-foreground">{order.customer}</span>
                <span className={`w-20 ${order.color}`}>{order.status}</span>
                <span className="w-20 text-right font-medium">{order.amount}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function TemplatesShowcase() {
  const [activeTab, setActiveTab] = useState<TemplateTab>("solutions");

  const urlMap: Record<TemplateTab, string> = {
    solutions: "scicom-design-hub.vercel.app/templates",
    login: "scicom-design-hub.vercel.app/login",
    dashboard: "scicom-design-hub.vercel.app/dashboard",
  };

  return (
    <section className="border-t border-border py-20">
      <div className="container mx-auto px-4 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={defaultViewport}
          variants={fadeIn}
          transition={{ duration: 0.5 }}
          className="mx-auto mb-8 max-w-2xl text-center"
        >
          <h2 className="mb-4 text-3xl font-bold">Templates &amp; Solutions</h2>
          <p className="text-muted-foreground">
            Pre-built page templates showcasing our design system and components in action.
          </p>
        </motion.div>

        {/* Tab buttons */}
        <div className="mb-6 flex justify-center">
          <div className="inline-flex rounded-full border border-border bg-muted/50 p-1">
            {templateTabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 rounded-full px-5 py-2 text-sm font-medium transition-all ${
                    activeTab === tab.id
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={defaultViewport}
          variants={fadeIn}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-lg">
            {/* Browser chrome */}
            <div className="flex items-center gap-2 border-b border-border bg-muted/50 px-4 py-3">
              <div className="flex gap-1.5">
                <div className="h-3 w-3 rounded-full bg-destructive/60" />
                <div className="h-3 w-3 rounded-full bg-yellow-500/60" />
                <div className="h-3 w-3 rounded-full bg-green-500/60" />
              </div>
              <div className="ml-4 flex-1 rounded-md bg-background/80 px-4 py-1 text-xs text-muted-foreground">
                {urlMap[activeTab]}
              </div>
            </div>

            {/* Preview content */}
            {activeTab === "solutions" && <SolutionsPreview />}
            {activeTab === "login" && <LoginPreview />}
            {activeTab === "dashboard" && <DashboardPreview />}
          </div>
        </motion.div>

        <div className="mt-8 text-center">
          <Link href="/templates">
            <Button variant="outline" size="lg" className="rounded-full px-8">
              Explore Templates
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <PageHeader />

      <main>
        {/* Hero Section with Carousel Background */}
        <section className="relative min-h-[600px] overflow-hidden lg:min-h-[700px]">
          {/* Layer 1: Background Images (light/dark) */}
          <Image
            src="/images/hero-bg-light.jpg"
            alt=""
            fill
            priority
            className="object-cover dark:hidden"
          />
          <Image
            src="/images/hero-bg.jpg"
            alt=""
            fill
            priority
            className="hidden object-cover dark:block"
          />

          {/* Layer 2: Overlay for text readability */}
          <div className="absolute inset-0 pointer-events-none bg-white/40 dark:bg-black/40" />

          {/* Layer 3: Hero Content on top */}
          <div className="relative z-10 flex min-h-[600px] items-center lg:min-h-[700px]">
          <div className="container mx-auto px-4 lg:px-8">
            <motion.div
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="mx-auto max-w-4xl text-center"
            >
              {/* Badge */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5 }}
                className="mb-6 inline-flex items-center gap-2 rounded-full border border-foreground/20 bg-background/60 px-4 py-2 text-sm font-medium text-foreground shadow-lg backdrop-blur-sm"
              >
                <Image
                  src="/images/scicom-logo.png"
                  alt="Scicom"
                  width={80}
                  height={24}
                  className="h-5 w-auto dark:brightness-0 dark:invert"
                />
                Design Hub
              </motion.div>

              {/* Main Heading */}
              <motion.h1
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="mb-6 text-4xl font-bold tracking-tight text-foreground drop-shadow-sm sm:text-5xl lg:text-6xl"
              >
                Beautiful{" "}
                <span className="text-primary">Component Library</span>
                <br />
                for Modern Web Apps
              </motion.h1>

              {/* Subtitle */}
              <motion.p
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="mx-auto mb-10 max-w-2xl text-lg text-muted-foreground lg:text-xl"
              >
                Bring your ideas to life with our comprehensive design system.
                Crafted for UI/UX developers, product engineers, entrepreneurs
                and visionaries.
              </motion.p>

              {/* CTA Buttons */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="flex flex-col items-center justify-center gap-4 sm:flex-row"
              >
                <Link href="/components">
                  <Button size="lg" className="rounded-full px-8 shadow-lg">
                    Browse Components
                  </Button>
                </Link>
                <Link href="/design-system">
                  <Button
                    size="lg"
                    variant="outline"
                    className="rounded-full border-foreground/20 bg-background/60 px-8 text-foreground shadow-lg backdrop-blur-sm hover:bg-background/80"
                  >
                    Explore Design System
                  </Button>
                </Link>
              </motion.div>
            </motion.div>
          </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="border-t border-border bg-muted/30 py-20">
          <div className="container mx-auto px-4 lg:px-8">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={fadeIn}
              transition={{ duration: 0.5 }}
              className="mx-auto mb-12 max-w-2xl text-center"
            >
              <h2 className="mb-4 text-3xl font-bold">
                Everything you need to build
              </h2>
              <p className="text-muted-foreground">
                A complete design system with colors, typography, and components
                extracted from your Figma design.
              </p>
            </motion.div>

            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="grid gap-8 md:grid-cols-3"
            >
              {/* Feature 1 — Design Tokens */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5 }}
                whileHover={{ y: -6, transition: { duration: 0.2 } }}
                className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-shadow hover:shadow-xl"
              >
                <div className="p-6 pb-4">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                    <Palette className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="mb-2 text-lg font-semibold">Design Tokens</h3>
                  <p className="text-sm text-muted-foreground">
                    Complete color palettes, typography scales, and spacing values
                    extracted directly from your Figma file.
                  </p>
                  <Link href="/design-system" className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline">
                    Explore tokens <ChevronRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
                <div className="relative mx-4 mb-4 overflow-hidden rounded-xl border border-border bg-muted/50">
                  <Image
                    src="/images/hero-bg-light.jpg"
                    alt="Design tokens preview"
                    width={600}
                    height={400}
                    className="h-52 w-full object-cover transition-transform duration-500 group-hover:scale-105 dark:hidden"
                  />
                  <Image
                    src="/images/hero-bg.jpg"
                    alt="Design tokens preview"
                    width={600}
                    height={400}
                    className="hidden h-52 w-full object-cover transition-transform duration-500 group-hover:scale-105 dark:block"
                  />
                </div>
              </motion.div>

              {/* Feature 2 — UI Components */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.1 }}
                whileHover={{ y: -6, transition: { duration: 0.2 } }}
                className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-shadow hover:shadow-xl"
              >
                <div className="p-6 pb-4">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                    <Component className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="mb-2 text-lg font-semibold">UI Components</h3>
                  <p className="text-sm text-muted-foreground">
                    Pre-built, accessible components powered by shadcn/ui and
                    styled with your design system.
                  </p>
                  <Link href="/components" className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline">
                    Browse components <ChevronRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
                <div className="relative mx-4 mb-4 overflow-hidden rounded-xl border border-border bg-muted/50">
                  <Image
                    src="/images/hero-bg-light.jpg"
                    alt="UI components preview"
                    width={600}
                    height={400}
                    className="h-52 w-full object-cover transition-transform duration-500 group-hover:scale-105 dark:hidden"
                  />
                  <Image
                    src="/images/hero-bg.jpg"
                    alt="UI components preview"
                    width={600}
                    height={400}
                    className="hidden h-52 w-full object-cover transition-transform duration-500 group-hover:scale-105 dark:block"
                  />
                </div>
              </motion.div>

              {/* Feature 3 — Documentation */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.2 }}
                whileHover={{ y: -6, transition: { duration: 0.2 } }}
                className="group overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-shadow hover:shadow-xl"
              >
                <div className="p-6 pb-4">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                    <BookOpen className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="mb-2 text-lg font-semibold">Documentation</h3>
                  <p className="text-sm text-muted-foreground">
                    Comprehensive guides and examples to help your team build
                    consistent interfaces.
                  </p>
                  <Link href="/documentation" className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline">
                    Read docs <ChevronRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
                <div className="relative mx-4 mb-4 overflow-hidden rounded-xl border border-border bg-muted/50">
                  <Image
                    src="/images/hero-bg-light.jpg"
                    alt="Documentation preview"
                    width={600}
                    height={400}
                    className="h-52 w-full object-cover transition-transform duration-500 group-hover:scale-105 dark:hidden"
                  />
                  <Image
                    src="/images/hero-bg.jpg"
                    alt="Documentation preview"
                    width={600}
                    height={400}
                    className="hidden h-52 w-full object-cover transition-transform duration-500 group-hover:scale-105 dark:block"
                  />
                </div>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Templates Showcase Section */}
        <TemplatesShowcase />

        {/* Component Examples Section */}
        <section className="border-t border-border bg-muted/30 py-20">
          <div className="container mx-auto px-4 lg:px-8">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={fadeIn}
              transition={{ duration: 0.5 }}
              className="mx-auto mb-12 max-w-2xl text-center"
            >
              <h2 className="mb-4 text-3xl font-bold">
                Component Examples
              </h2>
              <p className="text-muted-foreground">
                Explore our comprehensive library of pre-built components ready for your next project.
              </p>
            </motion.div>

            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="grid gap-6 md:grid-cols-2 lg:grid-cols-3"
            >
              {/* Dashboard Card */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="overflow-hidden">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">$45,231.89</div>
                    <p className="text-xs text-muted-foreground">+20.1% from last month</p>
                    <Progress value={75} className="mt-3 h-1" />
                  </CardContent>
                </Card>
              </motion.div>

              {/* Chart Preview Card */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.1 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="overflow-hidden">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">Analytics</CardTitle>
                      <BarChart3 className="h-4 w-4 text-primary" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex h-[80px] items-end gap-2">
                      {[40, 65, 45, 80, 55, 70, 90].map((height, i) => (
                        <div
                          key={i}
                          className="flex-1 rounded-t bg-primary/80"
                          style={{ height: `${height}%` }}
                        />
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* User Card */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.2 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Team Members</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {[
                      { name: "Sarah Chen", role: "Designer", initials: "SC" },
                      { name: "Alex Kim", role: "Developer", initials: "AK" },
                      { name: "Jordan Lee", role: "Manager", initials: "JL" },
                    ].map((member, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <Avatar className="h-8 w-8">
                          <AvatarFallback className="bg-primary/10 text-xs text-primary">
                            {member.initials}
                          </AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <p className="text-sm font-medium">{member.name}</p>
                          <p className="text-xs text-muted-foreground">{member.role}</p>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Navigation Preview */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.3 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Navigation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1">
                      {[
                        { icon: BarChart3, label: "Dashboard", active: true },
                        { icon: User, label: "Profile", active: false },
                        { icon: Settings, label: "Settings", active: false },
                      ].map((item, i) => (
                        <div
                          key={i}
                          className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm ${
                            item.active
                              ? "bg-primary text-white"
                              : "text-muted-foreground hover:bg-muted"
                          }`}
                        >
                          <item.icon className="h-4 w-4" />
                          {item.label}
                          <ChevronRight className="ml-auto h-4 w-4" />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Form Elements */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.4 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Form Elements</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input placeholder="Search..." className="pl-9" />
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" className="flex-1">
                        Submit
                      </Button>
                      <Button size="sm" variant="outline" className="flex-1">
                        Cancel
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Notifications */}
              <motion.div
                variants={fadeIn}
                transition={{ duration: 0.5, delay: 0.5 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="overflow-hidden">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">Notifications</CardTitle>
                      <Bell className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {[
                      { title: "New message", desc: "You have a new message", type: "info" },
                      { title: "Task complete", desc: "Project deployed", type: "success" },
                    ].map((notif, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded-md border p-2"
                      >
                        <div
                          className={`mt-0.5 h-2 w-2 rounded-full ${
                            notif.type === "success" ? "bg-green-500" : "bg-primary"
                          }`}
                        />
                        <div>
                          <p className="text-xs font-medium">{notif.title}</p>
                          <p className="text-xs text-muted-foreground">{notif.desc}</p>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={fadeIn}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="mt-10 text-center"
            >
              <Link href="/components">
                <Button variant="outline" className="rounded-full px-8">
                  View All Components
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </motion.div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="border-t border-border py-16">
          <div className="container mx-auto px-4 lg:px-8">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="grid gap-8 text-center md:grid-cols-4"
            >
              {[
                { value: "5", label: "Color Palettes" },
                { value: "55+", label: "Color Tokens" },
                { value: "13", label: "UI Components" },
                { value: "2", label: "Font Families" },
              ].map((stat, index) => (
                <motion.div
                  key={stat.label}
                  variants={fadeIn}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="text-4xl font-bold text-primary">
                    {stat.value}
                  </div>
                  <div className="mt-1 text-sm text-muted-foreground">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="border-t border-border bg-gradient-to-b from-muted/50 to-background py-20">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={defaultViewport}
            variants={fadeIn}
            transition={{ duration: 0.5 }}
            className="container mx-auto px-4 text-center lg:px-8"
          >
            <h2 className="mb-4 text-3xl font-bold">Ready to get started?</h2>
            <p className="mx-auto mb-8 max-w-xl text-muted-foreground">
              Explore the design system, browse components, or dive into the
              documentation.
            </p>
            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Link href="/design-system">
                <Button size="lg" className="rounded-full px-8">
                  View Design System
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/documentation">
                <Button
                  size="lg"
                  variant="outline"
                  className="rounded-full px-8"
                >
                  Read Documentation
                </Button>
              </Link>
            </div>
          </motion.div>
        </section>
      </main>

      <PageFooter description="Built with Next.js, Tailwind CSS, and shadcn/ui. Design tokens from Figma." />
    </div>
  );
}
