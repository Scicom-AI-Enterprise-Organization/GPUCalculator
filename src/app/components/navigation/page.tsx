"use client";

import { Menu, Home, Users, Settings, ChevronRight, PanelTop, PanelLeft, ChevronsRight, LayoutList } from "lucide-react";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";

const navigationSidebarSections = [
  {
    items: [
      { label: "Navbars", href: "#navbars", icon: PanelTop },
      { label: "Sidebars", href: "#sidebars", icon: PanelLeft },
      { label: "Breadcrumbs", href: "#breadcrumbs", icon: ChevronsRight },
      { label: "Tabs", href: "#tabs", icon: LayoutList },
    ],
  },
];

export default function NavigationPage() {
  return (
    <TwoColumnLayout
      title="Navigation"
      sidebarSections={navigationSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Navigation components help users find their way through your application.
          Includes navbars, sidebars, breadcrumbs, and tabs.
        </p>
      </div>

      {/* Navbars */}
      <section id="navbars" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Navbars</h2>
        <p className="mb-6 text-muted-foreground">
          Top navigation bars for application headers.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Navbar" count={1}>
            <div className="w-full max-w-[240px] rounded-lg border border-border bg-card">
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  <div className="h-6 w-6 rounded bg-primary" />
                  <span className="text-sm font-medium">Logo</span>
                </div>
                <Menu className="h-5 w-5 text-muted-foreground" />
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="With Links" count={1}>
            <div className="w-full max-w-[280px] rounded-lg border border-border bg-card">
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-4">
                  <div className="h-6 w-6 rounded bg-primary" />
                  <nav className="flex gap-3 text-xs">
                    <span className="text-primary font-medium">Home</span>
                    <span className="text-muted-foreground">Products</span>
                    <span className="text-muted-foreground">About</span>
                  </nav>
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Sidebars */}
      <section id="sidebars" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Sidebars</h2>
        <p className="mb-6 text-muted-foreground">
          Vertical navigation for application layouts.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Sidebar" count={1}>
            <div className="w-28 rounded-lg border border-border bg-card p-3">
              <div className="space-y-1">
                <div className="flex items-center gap-2 rounded bg-primary/10 p-2 text-primary">
                  <Home className="h-4 w-4" />
                  <span className="text-xs font-medium">Home</span>
                </div>
                <div className="flex items-center gap-2 p-2 text-muted-foreground hover:bg-muted rounded">
                  <Users className="h-4 w-4" />
                  <span className="text-xs">Users</span>
                </div>
                <div className="flex items-center gap-2 p-2 text-muted-foreground hover:bg-muted rounded">
                  <Settings className="h-4 w-4" />
                  <span className="text-xs">Settings</span>
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Collapsed Sidebar" count={1}>
            <div className="w-12 rounded-lg border border-border bg-card p-2">
              <div className="space-y-1">
                <div className="flex items-center justify-center rounded bg-primary/10 p-2 text-primary">
                  <Home className="h-4 w-4" />
                </div>
                <div className="flex items-center justify-center p-2 text-muted-foreground hover:bg-muted rounded">
                  <Users className="h-4 w-4" />
                </div>
                <div className="flex items-center justify-center p-2 text-muted-foreground hover:bg-muted rounded">
                  <Settings className="h-4 w-4" />
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Breadcrumbs */}
      <section id="breadcrumbs" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Breadcrumbs</h2>
        <p className="mb-6 text-muted-foreground">
          Show the user's current location in the site hierarchy.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Breadcrumbs" count={1}>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground hover:text-foreground cursor-pointer">Home</span>
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground hover:text-foreground cursor-pointer">Products</span>
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium">Details</span>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Tabs */}
      <section id="tabs" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">Tabs</h2>
        <p className="mb-6 text-muted-foreground">
          Tab navigation for switching between content sections.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Pill Tabs" count={1}>
            <div className="flex rounded-lg bg-muted p-1">
              <button className="rounded-md bg-card px-4 py-1.5 text-sm font-medium shadow-sm">
                Tab 1
              </button>
              <button className="px-4 py-1.5 text-sm text-muted-foreground">
                Tab 2
              </button>
              <button className="px-4 py-1.5 text-sm text-muted-foreground">
                Tab 3
              </button>
            </div>
          </ComponentCard>

          <ComponentCard title="Underline Tabs" count={1}>
            <div className="flex border-b border-border">
              <button className="border-b-2 border-primary px-4 py-2 text-sm font-medium text-primary">
                Tab 1
              </button>
              <button className="px-4 py-2 text-sm text-muted-foreground">
                Tab 2
              </button>
              <button className="px-4 py-2 text-sm text-muted-foreground">
                Tab 3
              </button>
            </div>
          </ComponentCard>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
