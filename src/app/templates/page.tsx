"use client";

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
import { SectionHeader } from "@/components/section-header";
import {
  LayoutDashboard,
  Users,
  Settings,
  FileText,
  BarChart3,
  MessageSquare,
  ArrowRight,
  Eye,
  Mic,
  FileCode,
  Bot,
  Activity,
  BookOpen,
  Bird,
  Ticket,
  LogIn,
  Lock,
  UserPlus,
  Cpu,
} from "lucide-react";

const templatesSidebarSections = [
  {
    items: [
      { label: "Login", href: "#login", icon: LogIn },
      { label: "Dashboards", href: "#dashboards", icon: LayoutDashboard },
      { label: "AI Solutions", href: "#ai-solutions", icon: Cpu },
      { label: "Knowledge Base", href: "#kbms", icon: BookOpen },
      { label: "Scicom Blujay", href: "#blujay", icon: Bird },
      { label: "ITMS", href: "#itms", icon: Ticket },
    ],
  },
];

const loginTemplates = [
  {
    title: "Standard Login",
    description: "Clean and modern login form with email/password authentication and social sign-in options.",
    icon: LogIn,
    features: ["Email & password", "Social auth buttons", "Remember me toggle", "Forgot password"],
    status: "Popular",
  },
  {
    title: "Registration",
    description: "User registration flow with multi-step form validation and progressive disclosure.",
    icon: UserPlus,
    features: ["Multi-step form", "Email verification", "Password strength", "Terms acceptance"],
    status: "New",
  },
  {
    title: "Secure Access",
    description: "Two-factor authentication and security-focused login with biometric support.",
    icon: Lock,
    features: ["2FA support", "Biometric login", "Security keys", "Session management"],
    status: null,
  },
];

const dashboardTemplates = [
  {
    title: "Analytics Dashboard",
    description: "Comprehensive analytics with interactive charts, KPIs, and real-time metrics for business intelligence.",
    icon: BarChart3,
    features: ["Interactive charts", "Real-time data", "Export reports", "Custom KPIs"],
    status: "Popular",
  },
  {
    title: "Admin Dashboard",
    description: "Full-featured admin panel with user management, settings, and system monitoring.",
    icon: LayoutDashboard,
    features: ["User management", "System health", "Activity logs", "Role-based access"],
    status: "New",
  },
  {
    title: "Team Dashboard",
    description: "Team collaboration workspace with task tracking, member activity, and project overview.",
    icon: Users,
    features: ["Task boards", "Team activity", "Project timeline", "Resource allocation"],
    status: null,
  },
];

const aiSolutionsTemplates = [
  {
    title: "Voice Agent",
    description: "AI-powered voice interaction system with real-time speech processing and natural conversation flow.",
    icon: Mic,
    features: ["Speech recognition", "Natural language processing", "Call management"],
    status: "New",
  },
  {
    title: "Agent Lobby",
    description: "Intelligent agent routing and queue management with availability tracking.",
    icon: Bot,
    features: ["Agent status", "Queue management", "Smart routing"],
    status: "Popular",
  },
  {
    title: "AI Monitoring",
    description: "Real-time monitoring and analytics for AI-powered contact center operations.",
    icon: Activity,
    features: ["Real-time metrics", "Alert system", "Performance tracking"],
    status: null,
  },
];

const kbmsTemplates = [
  {
    title: "Knowledge Base",
    description: "Centralized knowledge repository with search and categorization.",
    icon: FileText,
    features: ["Article management", "Category structure", "Full-text search"],
    status: "Popular",
  },
  {
    title: "Document Editor",
    description: "Rich text editor for creating and editing knowledge base articles.",
    icon: FileCode,
    features: ["WYSIWYG editor", "Version history", "Collaboration"],
    status: "New",
  },
  {
    title: "FAQ Management",
    description: "Frequently asked questions management with analytics.",
    icon: MessageSquare,
    features: ["FAQ categories", "Search analytics", "Auto-suggestions"],
    status: null,
  },
];

const blujayTemplates = [
  {
    title: "Workflow Builder",
    description: "Visual workflow designer for creating automated business processes.",
    icon: Settings,
    features: ["Drag & drop", "Conditional logic", "Integration support"],
    status: "Popular",
  },
  {
    title: "Task Management",
    description: "Task tracking and assignment system with priority management.",
    icon: Users,
    features: ["Task boards", "Assignments", "Due dates"],
    status: null,
  },
  {
    title: "Reports & Analytics",
    description: "Comprehensive reporting tools for business insights.",
    icon: BarChart3,
    features: ["Custom reports", "Dashboards", "Data export"],
    status: "New",
  },
];

const itmsTemplates = [
  {
    title: "Ticket Management",
    description: "IT service desk ticketing system with SLA tracking.",
    icon: MessageSquare,
    features: ["Ticket queue", "Priority levels", "SLA monitoring"],
    status: "Popular",
  },
  {
    title: "Asset Inventory",
    description: "IT asset management and tracking system.",
    icon: LayoutDashboard,
    features: ["Asset registry", "Lifecycle tracking", "Audit logs"],
    status: null,
  },
  {
    title: "Change Management",
    description: "IT change request and approval workflow system.",
    icon: Settings,
    features: ["Change requests", "Approval workflow", "Impact assessment"],
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
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          {status && (
            <Badge variant={status === "New" ? "default" : "secondary"}>
              {status}
            </Badge>
          )}
        </div>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
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

export default function TemplatesPage() {
  return (
    <TwoColumnLayout
      title="Templates"
      sidebarSections={templatesSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Pre-built page templates to accelerate your development. Each template
          is fully customizable and built with our design system components.
        </p>
      </div>

      {/* Login Templates */}
      <section id="login" className="mb-12">
        <SectionHeader
          title="Login"
          description="Authentication page templates with modern login flows, social sign-in, and security features."
          href="/templates/login"
        />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {loginTemplates.map((template) => (
            <TemplateCard key={template.title} {...template} />
          ))}
        </div>
      </section>

      {/* Dashboard Templates */}
      <section id="dashboards" className="mb-12">
        <SectionHeader
          title="Dashboards"
          description="Data-rich dashboard templates for analytics, admin panels, and team workspaces."
          href="/templates/dashboards"
        />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {dashboardTemplates.map((template) => (
            <TemplateCard key={template.title} {...template} />
          ))}
        </div>
      </section>

      {/* AI Solutions Templates */}
      <section id="ai-solutions" className="mb-12">
        <SectionHeader
          title="AI Solutions"
          description="AI-powered monitoring solutions for call centers and customer service automation."
          href="/templates/ai-solutions"
        />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {aiSolutionsTemplates.map((template) => (
            <TemplateCard key={template.title} {...template} />
          ))}
        </div>
      </section>

      {/* Knowledge Base Management System */}
      <section id="kbms" className="mb-12">
        <SectionHeader
          title="Knowledge Base Management System"
          description="Centralized knowledge management platform for documentation and self-service support."
          href="/templates/kbms"
        />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {kbmsTemplates.map((template) => (
            <TemplateCard key={template.title} {...template} />
          ))}
        </div>
      </section>

      {/* Scicom Blujay */}
      <section id="blujay" className="mb-12">
        <SectionHeader
          title="Scicom Blujay"
          description="Business process automation and workflow management platform."
          href="/templates/blujay"
        />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {blujayTemplates.map((template) => (
            <TemplateCard key={template.title} {...template} />
          ))}
        </div>
      </section>

      {/* ITMS */}
      <section id="itms" className="mb-0">
        <SectionHeader
          title="ITMS"
          description="IT Management System for service desk, asset management, and change control."
          href="/templates/itms"
        />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {itmsTemplates.map((template) => (
            <TemplateCard key={template.title} {...template} />
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
