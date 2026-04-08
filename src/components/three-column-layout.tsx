"use client";

import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { PageFooter } from "@/components/page-footer";
import {
  ChevronLeft,
  ChevronRight,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Menu,
  X,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SidebarItem {
  label: string;
  href: string;
}

interface SidebarSection {
  title?: string;
  items: SidebarItem[];
}

interface ThreeColumnLayoutProps {
  title: string;
  description?: string;
  backHref?: string;
  backLabel?: string;
  sidebarSections: SidebarSection[];
  documentationContent: React.ReactNode;
  children: React.ReactNode;
}

export function ThreeColumnLayout({
  title,
  description,
  backHref = "/components",
  backLabel = "Back to Components",
  sidebarSections,
  documentationContent,
  children,
}: ThreeColumnLayoutProps) {
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [mobileDocsOpen, setMobileDocsOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <PageHeader />

      {/* Mobile Header Bar */}
      <div className="sticky top-16 z-40 flex items-center justify-between border-b bg-background/95 px-4 py-2 backdrop-blur lg:hidden">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setMobileMenuOpen(true)}
          className="gap-2"
        >
          <Menu className="h-4 w-4" />
          <span>Menu</span>
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setMobileDocsOpen(true)}
          className="gap-2"
        >
          <span>Docs</span>
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {/* Mobile Left Sidebar Overlay */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="absolute left-0 top-0 h-full w-72 bg-background shadow-xl">
            <div className="flex items-center justify-between border-b p-4">
              <span className="font-semibold">Navigation</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setMobileMenuOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <nav className="space-y-6 p-4">
              <Link href={backHref} onClick={() => setMobileMenuOpen(false)}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="mb-4 -ml-2 text-muted-foreground hover:text-foreground"
                >
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  {backLabel}
                </Button>
              </Link>

              {sidebarSections.map((section, index) => (
                <div key={index}>
                  {section.title && (
                    <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {section.title}
                    </h3>
                  )}
                  <ul className="space-y-1">
                    {section.items.map((item) => (
                      <li key={item.href}>
                        <a
                          href={item.href}
                          onClick={() => setMobileMenuOpen(false)}
                          className="block rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                        >
                          {item.label}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </nav>
          </aside>
        </div>
      )}

      {/* Mobile Right Sidebar (Docs) Overlay */}
      {mobileDocsOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setMobileDocsOpen(false)}
          />
          <aside className="absolute right-0 top-0 h-full w-80 overflow-y-auto bg-background shadow-xl">
            <div className="flex items-center justify-between border-b p-4">
              <span className="font-semibold">Documentation</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setMobileDocsOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-4">{documentationContent}</div>
          </aside>
        </div>
      )}

      <div className="container mx-auto px-4 lg:px-8">
        <div className="flex gap-0 pb-8 lg:gap-6">
          {/* Left Column - Collapsible Sidebar Navigation */}
          <aside
            className={cn(
              "sticky top-16 hidden shrink-0 self-stretch border-r border-border transition-all duration-300 lg:block",
              leftSidebarOpen ? "w-56" : "w-12"
            )}
          >
            {/* Collapse Toggle Button */}
            <div className="flex justify-end pr-2 pt-4">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
                title={leftSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
              >
                {leftSidebarOpen ? (
                  <PanelLeftClose className="h-4 w-4" />
                ) : (
                  <PanelLeftOpen className="h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Sidebar Content */}
            <nav
              className={cn(
                "space-y-6 overflow-hidden pr-4 transition-all duration-300",
                leftSidebarOpen ? "opacity-100" : "opacity-0 pointer-events-none"
              )}
            >
              {/* Back Button */}
              <Link href={backHref}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="mb-4 -ml-2 text-muted-foreground hover:text-foreground"
                >
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  {backLabel}
                </Button>
              </Link>

              {sidebarSections.map((section, index) => (
                <div key={index}>
                  {section.title && (
                    <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {section.title}
                    </h3>
                  )}
                  <ul className="space-y-1">
                    {section.items.map((item) => (
                      <li key={item.href}>
                        <a
                          href={item.href}
                          className="block rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                        >
                          {item.label}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </nav>
          </aside>

          {/* Middle Column - Main Content */}
          <main className="min-w-0 flex-1 pt-8">
            <h1 className="mb-2 text-3xl font-bold tracking-tight md:text-4xl lg:text-5xl">
              {title}
            </h1>
            {description && (
              <p className="mb-8 text-base text-muted-foreground md:text-lg">
                {description}
              </p>
            )}
            {children}
          </main>

          {/* Right Column - Scrollable Documentation */}
          <aside
            className={cn(
              "sticky top-16 hidden shrink-0 self-start border-l border-border transition-all duration-300 lg:block",
              rightSidebarOpen ? "w-80" : "w-12"
            )}
            style={{ maxHeight: "calc(100vh - 4rem)" }}
          >
            {/* Collapse Toggle Button */}
            <div className="flex justify-start pl-2 pt-4">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
                title={rightSidebarOpen ? "Collapse docs" : "Expand docs"}
              >
                {rightSidebarOpen ? (
                  <PanelRightClose className="h-4 w-4" />
                ) : (
                  <PanelRightOpen className="h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Scrollable Documentation Content */}
            <div
              className={cn(
                "overflow-y-auto pl-6 pr-2 transition-all duration-300",
                rightSidebarOpen ? "opacity-100" : "opacity-0 pointer-events-none"
              )}
              style={{ maxHeight: "calc(100vh - 8rem)" }}
            >
              <div className="space-y-6 pb-8">
                <h3 className="text-lg font-semibold">Documentation</h3>
                {documentationContent}
              </div>
            </div>
          </aside>
        </div>
      </div>

      <PageFooter description="Scicom Design Hub - Component Documentation" />
    </div>
  );
}
