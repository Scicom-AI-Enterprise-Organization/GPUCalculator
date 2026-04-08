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
  Settings,
  Users,
  BarChart3,
  GitBranch,
  Workflow,
  ClipboardList,
  CheckSquare,
  Cog,
} from "lucide-react";

const blujaySidebarSections = [
  {
    items: [
      { label: "Workflow Builder", href: "#workflow-builder", icon: Workflow },
      { label: "Task Management", href: "#task-management", icon: CheckSquare },
      { label: "Reports & Analytics", href: "#reports", icon: BarChart3 },
      { label: "Process Automation", href: "#automation", icon: Cog },
      { label: "Team Collaboration", href: "#collaboration", icon: Users },
    ],
  },
];

const blujayTemplates = [
  {
    id: "workflow-builder",
    title: "Workflow Builder",
    description: "Visual workflow designer for creating automated business processes with drag-and-drop simplicity.",
    icon: Workflow,
    features: ["Drag & drop", "Conditional logic", "Integration support", "Template library", "Version control"],
    status: "Popular",
  },
  {
    id: "task-management",
    title: "Task Management",
    description: "Task tracking and assignment system with priority management and progress monitoring.",
    icon: ClipboardList,
    features: ["Task boards", "Assignments", "Due dates", "Priority levels", "Progress tracking"],
    status: null,
  },
  {
    id: "reports",
    title: "Reports & Analytics",
    description: "Comprehensive reporting tools for business insights and performance monitoring.",
    icon: BarChart3,
    features: ["Custom reports", "Dashboards", "Data export", "Scheduled reports", "KPI tracking"],
    status: "New",
  },
  {
    id: "automation",
    title: "Process Automation",
    description: "Automate repetitive tasks and business processes with intelligent rules and triggers.",
    icon: GitBranch,
    features: ["Rule engine", "Triggers", "Actions", "Scheduling", "Error handling"],
    status: null,
  },
  {
    id: "collaboration",
    title: "Team Collaboration",
    description: "Team workspace for collaboration, communication, and shared project management.",
    icon: Users,
    features: ["Team spaces", "Comments", "File sharing", "Notifications", "Activity feed"],
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

export default function BlujayPage() {
  return (
    <TwoColumnLayout
      title="Scicom Blujay"
      sidebarSections={blujaySidebarSections}
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
            <Workflow className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Scicom Blujay</h2>
            <p className="text-sm text-muted-foreground">Business process automation platform</p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          Business process automation and workflow management platform.
          Streamline operations, automate tasks, and improve team collaboration.
        </p>
      </div>

      <section className="mb-0">
        <div className="grid gap-6 md:grid-cols-2">
          {blujayTemplates.map((template) => (
            <div key={template.id} id={template.id}>
              <TemplateCard {...template} />
            </div>
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
