"use client";

import Link from "next/link";
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
} from "lucide-react";

const aimsSidebarSections = [
  {
    items: [
      { label: "Voice Agent", href: "#voice-agent", icon: Mic },
      { label: "Agent Lobby", href: "#agent-lobby", icon: Users },
      { label: "Analytics Dashboard", href: "#analytics", icon: BarChart3 },
      { label: "AI Chat Assistant", href: "#ai-chat", icon: MessageSquare },
      { label: "Smart Routing", href: "#smart-routing", icon: Route },
    ],
  },
];

const aimsTemplates = [
  {
    id: "voice-agent",
    title: "Voice Agent",
    description: "AI-powered voice interaction system with real-time speech processing and natural conversation flow. Perfect for customer service automation and voice-based interfaces.",
    icon: Mic,
    features: ["Speech recognition", "Natural language processing", "Call management", "Voice synthesis", "Multi-language support"],
    status: "New",
  },
  {
    id: "agent-lobby",
    title: "Agent Lobby",
    description: "Intelligent agent routing and queue management with availability tracking. Optimize your team's workload with smart assignment algorithms.",
    icon: Bot,
    features: ["Agent status", "Queue management", "Smart routing", "Performance metrics", "Real-time dashboard"],
    status: "Popular",
  },
  {
    id: "analytics",
    title: "Analytics Dashboard",
    description: "Comprehensive analytics with charts, KPIs, and real-time metrics for monitoring AI performance and call center operations.",
    icon: BarChart3,
    features: ["Real-time data", "Interactive charts", "Export reports", "Custom KPIs", "Trend analysis"],
    status: null,
  },
  {
    id: "ai-chat",
    title: "AI Chat Assistant",
    description: "Conversational AI interface for customer support and internal tools. Build intelligent chatbots with context awareness.",
    icon: MessageSquare,
    features: ["Context awareness", "Multi-turn conversations", "Knowledge base integration", "Sentiment analysis"],
    status: null,
  },
  {
    id: "smart-routing",
    title: "Smart Routing",
    description: "Intelligent call and ticket routing based on customer intent, agent skills, and availability patterns.",
    icon: Phone,
    features: ["Intent detection", "Skill-based routing", "Priority queues", "Overflow handling"],
    status: "New",
  },
];

interface TemplateCardProps {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  features: string[];
  status: string | null;
}

function TemplateCard({ title, description, icon: Icon, features, status }: TemplateCardProps) {
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
            <li key={feature} className="flex items-center gap-2 text-sm text-muted-foreground">
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

export default function AIMSPage() {
  return (
    <TwoColumnLayout
      title="AI Monitoring System"
      sidebarSections={aimsSidebarSections}
    >
      <div className="mb-8">
        <Link
          href="/templates"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Templates
        </Link>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <Activity className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">AI Monitoring System</h2>
            <p className="text-sm text-muted-foreground">Intelligent monitoring for call centers</p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          AI-powered monitoring solutions for call centers and customer service automation.
          Real-time analytics, agent assistance, and intelligent routing.
        </p>
      </div>

      <section className="mb-0">
        <div className="grid gap-6 md:grid-cols-2">
          {aimsTemplates.map((template) => (
            <div key={template.id} id={template.id}>
              <TemplateCard {...template} />
            </div>
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
