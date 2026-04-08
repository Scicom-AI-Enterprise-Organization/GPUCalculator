"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useBrand } from "@/components/brand-provider";
import { useThemeMode } from "@/hooks/use-theme-mode";
import {
  Search,
  Bell,
  Sun,
  Moon,
  BarChart3,
  MessageSquare,
  Phone,
  Mail,
  Users,
  AlertCircle,
} from "lucide-react";

// Agent status types
type AgentStatus = "negative" | "sla_exceeded" | "neutral" | "positive";

interface Agent {
  id: string;
  status: AgentStatus;
  duration: string;
}

// Generate mock agent data based on the image
const generateAgents = (): Agent[] => {
  const agents: Agent[] = [];

  // First row - 1 negative, 3 SLA exceeded, 4 neutral
  agents.push({ id: "#0001", status: "negative", duration: "00:00:00" });
  agents.push({ id: "#0001", status: "sla_exceeded", duration: "00:00:00" });
  agents.push({ id: "#0001", status: "sla_exceeded", duration: "00:00:00" });
  agents.push({ id: "#0001", status: "sla_exceeded", duration: "00:00:00" });
  for (let i = 0; i < 4; i++) {
    agents.push({ id: "#0001", status: "neutral", duration: "00:00:00" });
  }

  // Rest are mostly neutral with some positive at the end
  for (let i = 0; i < 16; i++) {
    agents.push({ id: "#0001", status: "neutral", duration: "00:00:00" });
  }

  // Last few positive
  for (let i = 0; i < 4; i++) {
    agents.push({ id: "#0001", status: "positive", duration: "00:00:00" });
  }

  return agents;
};

const agents = generateAgents();

// Status counts
const statusCounts = {
  overview: 32,
  negative: 1,
  sla_exceeded: 3,
  neutral: 25,
  positive: 3,
};

// Status color mapping
const getStatusColor = (status: AgentStatus) => {
  switch (status) {
    case "negative":
      return {
        bg: "bg-danger-500",
        text: "text-danger-500",
        label: "NEGATIVE",
        iconBg: "bg-danger-500",
        hasAlert: true,
      };
    case "sla_exceeded":
      return {
        bg: "bg-warning-500",
        text: "text-warning-600",
        label: "SLA EXCEEDED",
        iconBg: "bg-warning-500",
        hasAlert: false,
      };
    case "neutral":
      return {
        bg: "bg-primary",
        text: "text-primary",
        label: "NEUTRAL",
        iconBg: "bg-primary",
        hasAlert: false,
      };
    case "positive":
      return {
        bg: "bg-success-500",
        text: "text-success-500",
        label: "POSITIVE",
        iconBg: "bg-success-500",
        hasAlert: false,
      };
  }
};

// Agent Card Component
function AgentCard({ agent }: { agent: Agent }) {
  const statusConfig = getStatusColor(agent.status);

  return (
    <div className="flex flex-col items-center rounded-lg border border-border bg-card p-4 transition-all hover:shadow-md">
      {/* Avatar */}
      <div className="relative mb-3">
        <Avatar className={`h-16 w-16 border-2 ${statusConfig.hasAlert ? 'border-danger-500' : 'border-transparent'}`}>
          <AvatarFallback className={`${statusConfig.iconBg} text-white`}>
            {statusConfig.hasAlert ? (
              <AlertCircle className="h-6 w-6" />
            ) : (
              <Users className="h-6 w-6" />
            )}
          </AvatarFallback>
        </Avatar>
      </div>

      {/* Agent ID */}
      <span className="mb-1 text-sm font-medium text-foreground">{agent.id}</span>

      {/* Status */}
      <span className={`mb-1 text-xs font-bold uppercase ${statusConfig.text}`}>
        {statusConfig.label}
      </span>

      {/* Duration */}
      <span className="text-xs text-muted-foreground">{agent.duration}</span>
    </div>
  );
}

// Tab Component
function StatusTab({
  label,
  count,
  isActive,
  colorClass,
  onClick
}: {
  label: string;
  count: number;
  isActive: boolean;
  colorClass?: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
        isActive
          ? "border-primary text-primary"
          : "border-transparent text-muted-foreground hover:text-foreground"
      }`}
    >
      {label}
      <Badge
        variant="secondary"
        className={`h-5 min-w-[20px] rounded-full px-1.5 text-xs ${colorClass || ""}`}
      >
        {count}
      </Badge>
    </button>
  );
}

// Left Sidebar Navigation
function LeftSidebar() {
  const sidebarItems = [
    { icon: BarChart3, label: "Analytics", active: true },
    { icon: MessageSquare, label: "Messages", active: false },
    { icon: Phone, label: "Calls", active: false },
    { icon: Mail, label: "Email", active: false },
    { icon: Users, label: "Agents", active: false },
  ];

  return (
    <aside className="flex w-16 flex-col items-center border-r border-border bg-card py-4">
      {/* Logo */}
      <div className="mb-6 flex h-10 w-10 items-center justify-center rounded-lg">
        <span className="text-2xl font-bold text-primary">S</span>
      </div>

      {/* Navigation Items */}
      <nav className="flex flex-1 flex-col gap-2">
        {sidebarItems.map((item, index) => (
          <button
            key={index}
            className={`flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
              item.active
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            }`}
          >
            <item.icon className="h-5 w-5" />
          </button>
        ))}
      </nav>
    </aside>
  );
}

// Top Header
function TopHeader() {
  const { isDark, toggleTheme } = useThemeMode();

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-4">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm">
        <span className="text-foreground">Autonomous Agents</span>
        <span className="text-muted-foreground">/</span>
        <span className="text-foreground">Call Monitoring</span>
        <span className="text-muted-foreground">/</span>
        <span className="text-muted-foreground">Overview</span>
      </nav>

      {/* Right Side */}
      <div className="flex items-center gap-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search"
            className="h-9 w-48 pl-9 text-sm"
          />
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <Bell className="h-4 w-4" />
        </Button>

        {/* Theme Toggle */}
        <Button variant="ghost" size="icon" className="h-9 w-9" onClick={toggleTheme}>
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        {/* User Avatar */}
        <Avatar className="h-9 w-9">
          <AvatarFallback className="bg-muted text-foreground text-xs">DU</AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
}

export default function AgentLobbyPage() {
  const [activeTab, setActiveTab] = useState<string>("overview");

  // Filter agents based on active tab
  const filteredAgents = activeTab === "overview"
    ? agents
    : agents.filter(agent => agent.status === activeTab);

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Top Header - Full Width */}
      <TopHeader />

      {/* Content Area with Left Sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          {/* Status Tabs */}
          <div className="border-b border-border bg-card px-6">
            <div className="flex">
              <StatusTab
                label="Overview"
                count={statusCounts.overview}
                isActive={activeTab === "overview"}
                colorClass="bg-primary/20 text-primary"
                onClick={() => setActiveTab("overview")}
              />
              <StatusTab
                label="Negative"
                count={statusCounts.negative}
                isActive={activeTab === "negative"}
                colorClass="bg-danger-100 text-danger-600 dark:bg-danger-900/30 dark:text-danger-400"
                onClick={() => setActiveTab("negative")}
              />
              <StatusTab
                label="SLA Exceeded"
                count={statusCounts.sla_exceeded}
                isActive={activeTab === "sla_exceeded"}
                colorClass="bg-warning-100 text-warning-600 dark:bg-warning-900/30 dark:text-warning-400"
                onClick={() => setActiveTab("sla_exceeded")}
              />
              <StatusTab
                label="Neutral"
                count={statusCounts.neutral}
                isActive={activeTab === "neutral"}
                colorClass="bg-primary/20 text-primary"
                onClick={() => setActiveTab("neutral")}
              />
              <StatusTab
                label="Positive"
                count={statusCounts.positive}
                isActive={activeTab === "positive"}
                colorClass="bg-success-100 text-success-600 dark:bg-success-900/30 dark:text-success-400"
                onClick={() => setActiveTab("positive")}
              />
            </div>
          </div>

          {/* Agent Grid */}
          <div className="p-6">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8">
              {filteredAgents.map((agent, index) => (
                <AgentCard key={index} agent={agent} />
              ))}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
