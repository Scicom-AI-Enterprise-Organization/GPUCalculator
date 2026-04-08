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
  Mic,
  Bot,
  BarChart3,
  MessageSquare,
  Phone,
  Activity,
  Users,
  Route,
  Cpu,
  Settings,
  Headphones,
} from "lucide-react";

const aiSolutionsSidebarSections = [
  {
    items: [
      { label: "Voice Agent", href: "#voice-agent", icon: Mic },
      { label: "Agent Lobby", href: "#agent-lobby", icon: Users },
      { label: "AI Monitoring", href: "#ai-monitoring", icon: Activity },
      { label: "AI Chat Assistant", href: "#ai-chat", icon: MessageSquare },
      { label: "Smart Routing", href: "#smart-routing", icon: Route },
    ],
  },
];

const aiSolutionsTemplates = [
  {
    id: "voice-agent",
    title: "Voice Agent",
    description:
      "AI-powered voice interaction system with real-time speech processing and natural conversation flow. Perfect for customer service automation and voice-based interfaces.",
    icon: Mic,
    features: [
      "Speech recognition",
      "Natural language processing",
      "Call management",
      "Voice synthesis",
      "Multi-language support",
    ],
    status: "New",
  },
  {
    id: "agent-lobby",
    title: "Agent Lobby",
    description:
      "Intelligent agent routing and queue management with availability tracking. Optimize your team's workload with smart assignment algorithms.",
    icon: Bot,
    features: [
      "Agent status",
      "Queue management",
      "Smart routing",
      "Performance metrics",
      "Real-time dashboard",
    ],
    status: "Popular",
  },
  {
    id: "ai-monitoring",
    title: "AI Monitoring",
    description:
      "Real-time monitoring and analytics for AI-powered contact center operations. Track performance, detect anomalies, and optimize workflows.",
    icon: Activity,
    features: [
      "Real-time metrics",
      "Alert system",
      "Performance tracking",
      "Anomaly detection",
      "SLA monitoring",
    ],
    status: null,
  },
  {
    id: "ai-chat",
    title: "AI Chat Assistant",
    description:
      "Conversational AI interface for customer support and internal tools. Build intelligent chatbots with context awareness.",
    icon: MessageSquare,
    features: [
      "Context awareness",
      "Multi-turn conversations",
      "Knowledge base integration",
      "Sentiment analysis",
    ],
    status: null,
  },
  {
    id: "smart-routing",
    title: "Smart Routing",
    description:
      "Intelligent call and ticket routing based on customer intent, agent skills, and availability patterns.",
    icon: Phone,
    features: [
      "Intent detection",
      "Skill-based routing",
      "Priority queues",
      "Overflow handling",
    ],
    status: "New",
  },
];

/* ---------- Solutions page data ---------- */
const solutionsPages: Record<
  string,
  {
    title: string;
    desc: string;
    stats: { label: string; value: string }[];
    bars: number[];
  }
> = {
  "AI Monitoring": {
    title: "AI Monitoring System",
    desc: "Real-time monitoring and analytics for your AI-powered contact center.",
    stats: [
      { label: "Active Sessions", value: "1,247" },
      { label: "Avg Response", value: "1.2s" },
      { label: "Satisfaction", value: "94.8%" },
    ],
    bars: [70, 85, 60, 90, 45, 80, 55, 95],
  },
  "Voice Agent": {
    title: "Voice Agent Console",
    desc: "Manage and configure voice-based AI agents for customer interactions.",
    stats: [
      { label: "Active Calls", value: "342" },
      { label: "Avg Duration", value: "4.5m" },
      { label: "Resolution", value: "87.2%" },
    ],
    bars: [60, 75, 80, 50, 90, 65, 70, 85],
  },
  "Agent Lobby": {
    title: "Agent Lobby",
    desc: "Monitor agent availability and manage queue assignments in real-time.",
    stats: [
      { label: "Online Agents", value: "58" },
      { label: "Queue Depth", value: "23" },
      { label: "Utilization", value: "76.4%" },
    ],
    bars: [50, 65, 45, 80, 70, 55, 90, 60],
  },
  Analytics: {
    title: "Analytics Overview",
    desc: "Comprehensive analytics dashboard for AI performance insights.",
    stats: [
      { label: "Conversations", value: "12.4k" },
      { label: "Accuracy", value: "96.1%" },
      { label: "Cost Saved", value: "$42.8k" },
    ],
    bars: [80, 90, 70, 85, 95, 60, 75, 88],
  },
};

/* ---------- Interactive AI Solutions Preview ---------- */
function AISolutionsPreviewDemo() {
  const [activePage, setActivePage] = useState("AI Monitoring");
  const pages = Object.keys(solutionsPages);
  const current = solutionsPages[activePage];

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
          scicom-design-hub.vercel.app/ai-solutions
        </div>
      </div>

      <div className="flex h-[560px] bg-background">
        {/* Sidebar */}
        <div className="w-56 shrink-0 border-r border-border bg-card p-4">
          <div className="mb-4 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20">
              <Cpu className="h-4 w-4 text-primary" />
            </div>
            <div>
              <div className="text-sm font-semibold text-foreground">
                Solutions
              </div>
              <div className="text-[10px] text-muted-foreground">v1.0.1</div>
            </div>
          </div>

          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Contact Center
          </div>
          {pages.slice(0, 4).map((page) => (
            <div
              key={page}
              onClick={() => setActivePage(page)}
              className={`mb-0.5 cursor-pointer rounded-md px-3 py-2 text-sm transition-colors ${
                activePage === page
                  ? "bg-primary/10 font-medium text-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              {page}
            </div>
          ))}

          <div className="mb-2 mt-4 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Management
          </div>
          {["Knowledge Base", "Ticketing", "Blujay CRM"].map((item) => (
            <div
              key={item}
              className="mb-0.5 cursor-pointer rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {item}
            </div>
          ))}

          <div className="mb-2 mt-4 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Resources
          </div>
          {["API Docs", "Changelog"].map((item) => (
            <div
              key={item}
              className="mb-0.5 cursor-pointer rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {item}
            </div>
          ))}
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Breadcrumb */}
          <div className="mb-4 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Settings className="h-3 w-3" />
            Contact Center
            <span className="mx-1">›</span>
            <span className="text-foreground">{activePage}</span>
          </div>

          <motion.div
            key={activePage}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="mb-1 text-xl font-semibold text-foreground">
              {current.title}
            </h3>
            <p className="mb-5 text-sm text-muted-foreground">{current.desc}</p>

            {/* Stat Cards */}
            <div className="mb-5 grid grid-cols-3 gap-3">
              {current.stats.map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="rounded-lg border border-border bg-card p-3 text-center"
                >
                  <div className="text-[10px] text-muted-foreground">
                    {stat.label}
                  </div>
                  <div className="text-2xl font-bold text-foreground">
                    {stat.value}
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Session Timeline Chart */}
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">
                  Session Timeline
                </span>
                <div className="flex items-center gap-1.5">
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  <span className="text-[10px] text-muted-foreground">
                    Live
                  </span>
                </div>
              </div>
              <div className="flex h-36 items-end gap-1.5">
                {current.bars.map((value, i) => (
                  <motion.div
                    key={`${activePage}-${i}`}
                    initial={{ height: 0 }}
                    animate={{ height: `${value}%` }}
                    transition={{ delay: i * 0.05, duration: 0.4 }}
                    className="flex-1 rounded-t bg-primary/70 transition-colors hover:bg-primary"
                  />
                ))}
              </div>
            </div>

            {/* Progress Bars */}
            <div className="mt-4 rounded-lg border border-border bg-card p-4">
              <span className="mb-3 block text-sm font-medium text-foreground">
                Performance Metrics
              </span>
              {[
                { label: "Response Time", value: 82 },
                { label: "Resolution Rate", value: 91 },
                { label: "Customer Satisfaction", value: 88 },
              ].map((metric) => (
                <div key={metric.label} className="mb-3 last:mb-0">
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {metric.label}
                    </span>
                    <span className="font-medium text-foreground">
                      {metric.value}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${metric.value}%` }}
                      transition={{ delay: 0.5, duration: 0.6 }}
                      className="h-full rounded-full bg-primary"
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
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

export default function AISolutionsPage() {
  return (
    <TwoColumnLayout
      title="AI Solutions"
      sidebarSections={aiSolutionsSidebarSections}
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
            <Cpu className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">AI Solutions Templates</h2>
            <p className="text-sm text-muted-foreground">
              AI-powered contact center & monitoring
            </p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          AI-powered monitoring solutions for call centers and customer service
          automation. Real-time analytics, agent assistance, and intelligent
          routing templates.
        </p>
      </div>

      {/* Interactive Preview */}
      <section className="mb-12">
        <h3 className="mb-2 text-xl font-semibold">Live Preview</h3>
        <p className="mb-4 text-sm text-muted-foreground">
          Interactive preview of the AI solutions interface. Click sidebar items
          to switch between different views.
        </p>
        <AISolutionsPreviewDemo />
      </section>

      {/* Template Cards */}
      <section className="mb-0">
        <h3 className="mb-6 text-xl font-semibold">
          All AI Solutions Templates
        </h3>
        <div className="grid gap-6 md:grid-cols-2">
          {aiSolutionsTemplates.map((template) => (
            <div key={template.id} id={template.id}>
              <TemplateCard {...template} />
            </div>
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
