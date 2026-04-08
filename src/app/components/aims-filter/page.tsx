"use client";

import { TwoColumnLayout } from "@/components/two-column-layout";
import { AIMSFilter } from "@/components/aims-filter";
import { Eye, Filter, SlidersHorizontal, BookOpen } from "lucide-react";

const aimsFilterSidebarSections = [
  {
    items: [
      { label: "Overview", href: "#overview", icon: Eye },
      { label: "Standard Mode", href: "#standard-mode", icon: Filter },
      { label: "Advanced Mode", href: "#advanced-mode", icon: SlidersHorizontal },
      { label: "Usage", href: "#usage", icon: BookOpen },
    ],
  },
];

export default function AIMSFilterPage() {
  return (
    <TwoColumnLayout
      title="AIMS Filter"
      sidebarSections={aimsFilterSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          A comprehensive filter panel designed for the AIMS platform. Supports Standard and Advanced
          modes for filtering sessions, users, time ranges, sentiment, and more.
        </p>
      </div>

      {/* Overview */}
      <section id="overview" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Overview</h2>
        <p className="mb-6 text-muted-foreground">
          The AIMS Filter component provides a flexible filtering interface with collapsible sections
          and toggle switches for each filter option.
        </p>
        <div className="flex items-center justify-center rounded-xl border border-border bg-muted/50 p-12 shadow-sm md:p-16">
          <AIMSFilter />
        </div>
      </section>

      {/* Standard Mode */}
      <section id="standard-mode" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Standard Mode</h2>
        <p className="mb-6 text-muted-foreground">
          Standard mode includes a global search field and filter sections for Users, Time, Sentiment, and Language.
        </p>
        <div className="rounded-lg border bg-muted/50 p-6">
          <h3 className="mb-4 font-medium">Filter Sections</h3>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li><strong>Users:</strong> Filter by VIP status</li>
            <li><strong>Time:</strong> Filter by session duration (&lt; 1, 2, or 3 minutes)</li>
            <li><strong>Sentiment:</strong> Filter by positive, negative, or neutral sentiment</li>
            <li><strong>Language:</strong> Filter by language (English)</li>
          </ul>
        </div>
      </section>

      {/* Advanced Mode */}
      <section id="advanced-mode" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Advanced Mode</h2>
        <p className="mb-6 text-muted-foreground">
          Advanced mode provides additional filtering options for Value Stream and Category Type.
        </p>
        <div className="rounded-lg border bg-muted/50 p-6">
          <h3 className="mb-4 font-medium">Filter Sections</h3>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li><strong>Users:</strong> Filter by VIP status</li>
            <li><strong>Value Stream:</strong> Filter by Assurance, Billing, or Fulfillment</li>
            <li><strong>Category Type:</strong> Filter by Enquiry, Negative, or Neutral</li>
          </ul>
        </div>
      </section>

      {/* Usage */}
      <section id="usage" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">Usage</h2>
        <p className="mb-6 text-muted-foreground">
          Import and use the AIMSFilter component in your application.
        </p>
        <div className="rounded-lg border bg-background p-4 text-sm">
          <pre className="text-foreground">
{`import { AIMSFilter } from "@/components/aims-filter";

export function MyPage() {
  return (
    <AIMSFilter
      onFilterChange={(filters) => {
        console.log("Filters changed:", filters);
      }}
    />
  );
}`}
          </pre>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
