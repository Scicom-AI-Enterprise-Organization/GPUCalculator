"use client";

import { PageHeader } from "@/components/page-header";
import { PageFooter } from "@/components/page-footer";
import { PageSidebar } from "@/components/page-sidebar";

interface NavItem {
  label: string;
  href: string;
}

interface SidebarSection {
  title?: string;
  items: NavItem[];
}

interface TwoColumnLayoutProps {
  /** Page title displayed in the header area */
  title: string;
  /** Navigation sections for sidebar */
  sidebarSections: SidebarSection[];
  /** Toggle options for sidebar (e.g., Product / Editorial) */
  toggleOptions?: {
    options: string[];
    defaultValue?: string;
    onChange?: (value: string) => void;
  };
  /** Current active toggle value */
  activeToggle?: string;
  /** Main content */
  children: React.ReactNode;
}

/**
 * Two-column page layout with sidebar navigation.
 * Used for Components, Documentation, and Design System pages.
 */
export function TwoColumnLayout({
  title,
  sidebarSections,
  toggleOptions,
  activeToggle,
  children,
}: TwoColumnLayoutProps) {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
      <PageHeader />

      <div className="flex min-h-0 flex-1">
        {/* Left Column - Sidebar Navigation (static) */}
        <PageSidebar
          sections={sidebarSections}
          toggleOptions={toggleOptions}
          activeToggle={activeToggle}
        />

        {/* Right Column - Scrollable Content */}
        <main className="min-w-0 flex-1 overflow-y-auto px-4 pb-8 pt-8 lg:px-8">
          {/* Page Title */}
          <h1 className="mb-8 text-5xl font-bold uppercase tracking-tight lg:text-6xl">
            {title}
          </h1>

          {/* Content */}
          {children}

          {/* Footer inside scrollable area */}
          <PageFooter />
        </main>
      </div>
    </div>
  );
}
