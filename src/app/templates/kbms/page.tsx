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
  FileText,
  FileCode,
  MessageSquare,
  Search,
  BookOpen,
  FolderTree,
  FileEdit,
  HelpCircle,
} from "lucide-react";

const kbmsSidebarSections = [
  {
    items: [
      { label: "Knowledge Base", href: "#knowledge-base", icon: BookOpen },
      { label: "Document Editor", href: "#document-editor", icon: FileEdit },
      { label: "FAQ Management", href: "#faq", icon: HelpCircle },
      { label: "Search Interface", href: "#search", icon: Search },
      { label: "Category Manager", href: "#categories", icon: FolderTree },
    ],
  },
];

const kbmsTemplates = [
  {
    id: "knowledge-base",
    title: "Knowledge Base",
    description: "Centralized knowledge repository with search and categorization. Organize articles, guides, and documentation in one place.",
    icon: BookOpen,
    features: ["Article management", "Category structure", "Full-text search", "Version control", "Access permissions"],
    status: "Popular",
  },
  {
    id: "document-editor",
    title: "Document Editor",
    description: "Rich text editor for creating and editing knowledge base articles with collaborative features.",
    icon: FileCode,
    features: ["WYSIWYG editor", "Version history", "Collaboration", "Media embedding", "Templates"],
    status: "New",
  },
  {
    id: "faq",
    title: "FAQ Management",
    description: "Frequently asked questions management system with analytics and auto-suggestions.",
    icon: MessageSquare,
    features: ["FAQ categories", "Search analytics", "Auto-suggestions", "Voting system", "Related questions"],
    status: null,
  },
  {
    id: "search",
    title: "Search Interface",
    description: "Advanced search interface with filters, suggestions, and instant results for finding knowledge quickly.",
    icon: Search,
    features: ["Instant search", "Filters", "Suggestions", "Search history", "Relevance ranking"],
    status: null,
  },
  {
    id: "categories",
    title: "Category Manager",
    description: "Hierarchical category and tag management for organizing knowledge base content.",
    icon: FolderTree,
    features: ["Tree structure", "Drag & drop", "Bulk operations", "Tag management", "Content mapping"],
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

export default function KBMSPage() {
  return (
    <TwoColumnLayout
      title="Knowledge Base Management"
      sidebarSections={kbmsSidebarSections}
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
            <BookOpen className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Knowledge Base Management System</h2>
            <p className="text-sm text-muted-foreground">Centralized documentation platform</p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          Comprehensive knowledge management platform for documentation and self-service support.
          Create, organize, and share knowledge across your organization.
        </p>
      </div>

      <section className="mb-0">
        <div className="grid gap-6 md:grid-cols-2">
          {kbmsTemplates.map((template) => (
            <div key={template.id} id={template.id}>
              <TemplateCard {...template} />
            </div>
          ))}
        </div>
      </section>
    </TwoColumnLayout>
  );
}
