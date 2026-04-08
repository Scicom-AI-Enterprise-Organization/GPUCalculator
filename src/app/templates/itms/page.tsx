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
  MessageSquare,
  LayoutDashboard,
  Settings,
  Server,
  Shield,
  FileCheck,
  Ticket,
  Package,
  GitBranch,
  ShoppingBag,
  AlertTriangle,
} from "lucide-react";

const itmsSidebarSections = [
  {
    items: [
      { label: "Ticket Management", href: "#ticket-management", icon: Ticket },
      { label: "Asset Inventory", href: "#asset-inventory", icon: Package },
      { label: "Change Management", href: "#change-management", icon: GitBranch },
      { label: "Service Catalog", href: "#service-catalog", icon: ShoppingBag },
      { label: "Incident Management", href: "#incident-management", icon: AlertTriangle },
    ],
  },
];

const itmsTemplates = [
  {
    id: "ticket-management",
    title: "Ticket Management",
    description: "IT service desk ticketing system with SLA tracking and priority management.",
    icon: MessageSquare,
    features: ["Ticket queue", "Priority levels", "SLA monitoring", "Assignment rules", "Status tracking"],
    status: "Popular",
  },
  {
    id: "asset-inventory",
    title: "Asset Inventory",
    description: "IT asset management and tracking system for hardware and software inventory.",
    icon: Server,
    features: ["Asset registry", "Lifecycle tracking", "Audit logs", "Depreciation", "Warranty tracking"],
    status: null,
  },
  {
    id: "change-management",
    title: "Change Management",
    description: "IT change request and approval workflow system with impact assessment.",
    icon: FileCheck,
    features: ["Change requests", "Approval workflow", "Impact assessment", "Risk analysis", "Rollback plans"],
    status: "New",
  },
  {
    id: "service-catalog",
    title: "Service Catalog",
    description: "Self-service IT service catalog for requesting IT services and support.",
    icon: LayoutDashboard,
    features: ["Service requests", "Catalog items", "Request forms", "Approval flows", "Fulfillment tracking"],
    status: null,
  },
  {
    id: "incident-management",
    title: "Incident Management",
    description: "IT incident tracking and resolution system with escalation procedures.",
    icon: Shield,
    features: ["Incident logging", "Escalation rules", "Resolution tracking", "Root cause analysis", "Knowledge linking"],
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

export default function ITMSPage() {
  return (
    <TwoColumnLayout
      title="ITMS"
      sidebarSections={itmsSidebarSections}
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
            <Server className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">IT Management System</h2>
            <p className="text-sm text-muted-foreground">IT service and asset management</p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          IT Management System for service desk, asset management, and change control.
          Streamline IT operations and improve service delivery.
        </p>
      </div>

      <section className="mb-0">
        <div className="grid gap-6 md:grid-cols-2">
          {itmsTemplates.map((template) => (
            <div key={template.id} id={template.id}>
              <TemplateCard {...template} />
            </div>
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
