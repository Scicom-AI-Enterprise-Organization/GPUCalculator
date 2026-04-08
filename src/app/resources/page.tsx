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
  FileText,
  Presentation,
  FileSpreadsheet,
  Download,
  Eye,
  FolderOpen,
  BookOpen,
  ClipboardList,
  FileCheck,
  Layers,
  Figma,
} from "lucide-react";

const resourcesSidebarSections = [
  {
    items: [
      { label: "Marketing Decks", href: "#marketing-decks", icon: Presentation },
      { label: "Documentation Templates", href: "#documentation-templates", icon: FileText },
      { label: "Design Assets", href: "#design-assets", icon: Figma },
    ],
  },
];

const marketingDecks = [
  {
    title: "Company Overview",
    description: "Corporate presentation with company history, services, and value proposition.",
    icon: Presentation,
    format: "PPTX",
    size: "12.5 MB",
    lastUpdated: "Jan 2026",
    status: "Updated",
  },
  {
    title: "Product Portfolio",
    description: "Comprehensive overview of all Scicom products and solutions.",
    icon: Presentation,
    format: "PPTX",
    size: "18.2 MB",
    lastUpdated: "Jan 2026",
    status: "New",
  },
  {
    title: "AIMS Presentation",
    description: "AI Monitoring System features, benefits, and case studies.",
    icon: Presentation,
    format: "PPTX",
    size: "8.7 MB",
    lastUpdated: "Dec 2025",
    status: null,
  },
  {
    title: "Client Success Stories",
    description: "Case studies and testimonials from satisfied clients.",
    icon: Presentation,
    format: "PPTX",
    size: "15.3 MB",
    lastUpdated: "Dec 2025",
    status: "Popular",
  },
];

const documentationTemplates = [
  {
    title: "User Requirements Specification (URS)",
    description: "Template for documenting user requirements and business needs.",
    icon: ClipboardList,
    format: "DOCX",
    size: "245 KB",
    lastUpdated: "Jan 2026",
    status: "Template",
  },
  {
    title: "Functional Requirements Specification (FRS)",
    description: "Template for detailed functional requirements documentation.",
    icon: FileCheck,
    format: "DOCX",
    size: "312 KB",
    lastUpdated: "Jan 2026",
    status: "Template",
  },
  {
    title: "IT Design Architecture (ITDA)",
    description: "Technical architecture documentation template for IT solutions.",
    icon: Layers,
    format: "DOCX",
    size: "428 KB",
    lastUpdated: "Dec 2025",
    status: "Template",
  },
  {
    title: "Test Plan Template",
    description: "Comprehensive test plan template for QA documentation.",
    icon: FileText,
    format: "DOCX",
    size: "198 KB",
    lastUpdated: "Nov 2025",
    status: null,
  },
  {
    title: "Project Charter",
    description: "Project initiation document template.",
    icon: BookOpen,
    format: "DOCX",
    size: "156 KB",
    lastUpdated: "Oct 2025",
    status: null,
  },
];

const designAssets = [
  {
    title: "Brand Guidelines",
    description: "Complete brand identity guidelines including logos, colors, and typography.",
    icon: FolderOpen,
    format: "PDF",
    size: "24.8 MB",
    lastUpdated: "Jan 2026",
    status: "Essential",
  },
  {
    title: "Logo Package",
    description: "All logo variants in various formats (SVG, PNG, EPS).",
    icon: FolderOpen,
    format: "ZIP",
    size: "8.2 MB",
    lastUpdated: "Jan 2026",
    status: null,
  },
  {
    title: "Icon Library",
    description: "Custom icon set for Scicom applications.",
    icon: Layers,
    format: "ZIP",
    size: "4.5 MB",
    lastUpdated: "Dec 2025",
    status: "New",
  },
  {
    title: "Presentation Templates",
    description: "PowerPoint and Google Slides templates with brand styling.",
    icon: Presentation,
    format: "ZIP",
    size: "6.8 MB",
    lastUpdated: "Dec 2025",
    status: null,
  },
];

interface ResourceCardProps {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  format: string;
  size: string;
  lastUpdated: string;
  status: string | null;
}

function ResourceCard({ title, description, icon: Icon, format, size, lastUpdated, status }: ResourceCardProps) {
  return (
    <Card className="h-full transition-all hover:-translate-y-1 hover:shadow-lg">
      <CardHeader className="pb-3">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div className="flex items-center gap-2">
            {status && (
              <Badge variant={status === "New" || status === "Updated" ? "default" : "secondary"}>
                {status}
              </Badge>
            )}
            <Badge variant="outline" className="text-xs">
              {format}
            </Badge>
          </div>
        </div>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription className="text-sm">{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Size: {size}</span>
          <span>Updated: {lastUpdated}</span>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="flex-1">
            <Eye className="mr-2 h-4 w-4" />
            Preview
          </Button>
          <Button size="sm" className="flex-1">
            <Download className="mr-2 h-4 w-4" />
            Download
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ResourcesPage() {
  return (
    <TwoColumnLayout
      title="Resources"
      sidebarSections={resourcesSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Marketing materials, documentation templates, and design assets.
          Download resources to help with sales presentations, project documentation,
          and brand compliance.
        </p>
      </div>

      {/* Marketing Decks */}
      <section id="marketing-decks" className="mb-12">
        <SectionHeader
          title="Marketing Decks"
          description="Sales presentations and marketing materials for client meetings and proposals."
        />
        <div className="grid gap-6 md:grid-cols-2">
          {marketingDecks.map((resource) => (
            <ResourceCard key={resource.title} {...resource} />
          ))}
        </div>
      </section>

      {/* Documentation Templates */}
      <section id="documentation-templates" className="mb-12">
        <SectionHeader
          title="Documentation Templates"
          description="Standard templates for URS, FRS, ITDA, and other project documentation."
        />
        <div className="grid gap-6 md:grid-cols-2">
          {documentationTemplates.map((resource) => (
            <ResourceCard key={resource.title} {...resource} />
          ))}
        </div>
      </section>

      {/* Design Assets */}
      <section id="design-assets" className="mb-0">
        <SectionHeader
          title="Design Assets"
          description="Brand guidelines, logos, icons, and other design resources."
        />
        <div className="grid gap-6 md:grid-cols-2">
          {designAssets.map((resource) => (
            <ResourceCard key={resource.title} {...resource} />
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
