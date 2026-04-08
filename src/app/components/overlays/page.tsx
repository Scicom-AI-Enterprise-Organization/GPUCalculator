"use client";

import { Button } from "@/components/ui/button";
import { X, AlertCircle, Check, Info, AppWindow, MessageCircle, MessageSquare, Bell } from "lucide-react";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";

const overlaysSidebarSections = [
  {
    items: [
      { label: "Modals & Dialogs", href: "#modals", icon: AppWindow },
      { label: "Tooltips", href: "#tooltips", icon: MessageCircle },
      { label: "Popovers", href: "#popovers", icon: MessageSquare },
      { label: "Toasts", href: "#toasts", icon: Bell },
    ],
  },
];

export default function OverlaysPage() {
  return (
    <TwoColumnLayout
      title="Overlays"
      sidebarSections={overlaysSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Overlay components that appear on top of other content. Includes modals,
          tooltips, popovers, and toast notifications.
        </p>
      </div>

      {/* Modals & Dialogs */}
      <section id="modals" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Modals & Dialogs</h2>
        <p className="mb-6 text-muted-foreground">
          Modal windows for focused interactions and confirmations.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Dialog" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-lg">
              <div className="flex items-center justify-between border-b border-border p-3">
                <span className="text-sm font-medium">Dialog Title</span>
                <X className="h-4 w-4 text-muted-foreground cursor-pointer hover:text-foreground" />
              </div>
              <div className="p-4">
                <p className="text-sm text-muted-foreground">
                  Are you sure you want to continue with this action?
                </p>
              </div>
              <div className="flex justify-end gap-2 border-t border-border p-3">
                <Button size="sm" variant="outline">
                  Cancel
                </Button>
                <Button size="sm">
                  Confirm
                </Button>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Confirmation Dialog" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-lg">
              <div className="p-4 text-center">
                <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-danger-100 dark:bg-danger-900/30">
                  <AlertCircle className="h-6 w-6 text-danger-600" />
                </div>
                <h3 className="text-sm font-medium">Delete Item?</h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  This action cannot be undone.
                </p>
              </div>
              <div className="flex gap-2 border-t border-border p-3">
                <Button size="sm" variant="outline" className="flex-1">
                  Cancel
                </Button>
                <Button size="sm" variant="destructive" className="flex-1">
                  Delete
                </Button>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Tooltips */}
      <section id="tooltips" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Tooltips</h2>
        <p className="mb-6 text-muted-foreground">
          Contextual hints that appear on hover or focus.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Tooltip" count={1}>
            <div className="flex flex-col items-center gap-3">
              <div className="relative rounded bg-background px-3 py-1.5 text-xs text-white">
                Tooltip text
                <div className="absolute -bottom-1 left-1/2 h-2 w-2 -translate-x-1/2 rotate-45 bg-background" />
              </div>
              <Button size="sm" variant="outline">
                Hover me
              </Button>
            </div>
          </ComponentCard>

          <ComponentCard title="Tooltip Positions" count={1}>
            <div className="flex flex-wrap gap-2 justify-center">
              <div className="rounded bg-background px-2 py-1 text-[10px] text-white">Top</div>
              <div className="rounded bg-background px-2 py-1 text-[10px] text-white">Bottom</div>
              <div className="rounded bg-background px-2 py-1 text-[10px] text-white">Left</div>
              <div className="rounded bg-background px-2 py-1 text-[10px] text-white">Right</div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Popovers */}
      <section id="popovers" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Popovers</h2>
        <p className="mb-6 text-muted-foreground">
          Rich content panels that appear next to trigger elements.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Info Popover" count={1}>
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card p-3 shadow-lg">
              <div className="flex items-center gap-2 mb-2">
                <Info className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Popover Title</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Additional information or context that helps the user understand.
              </p>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Toasts */}
      <section id="toasts" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">Toasts & Notifications</h2>
        <p className="mb-6 text-muted-foreground">
          Temporary messages that provide feedback about an action.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Success Toast" count={1}>
            <div className="w-full max-w-[220px] flex items-center gap-3 rounded-lg border border-border bg-card p-3 shadow-md">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-success-100 dark:bg-success-900/30">
                <Check className="h-4 w-4 text-success-600" />
              </div>
              <div>
                <div className="text-sm font-medium">Success!</div>
                <div className="text-xs text-muted-foreground">Changes saved successfully.</div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Error Toast" count={1}>
            <div className="w-full max-w-[220px] flex items-center gap-3 rounded-lg border border-danger-200 bg-danger-50 p-3 dark:border-danger-800 dark:bg-danger-900/30">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-danger-100 dark:bg-danger-900/50">
                <AlertCircle className="h-4 w-4 text-danger-600" />
              </div>
              <div>
                <div className="text-sm font-medium text-danger-700 dark:text-danger-400">Error</div>
                <div className="text-xs text-danger-600 dark:text-danger-300">Something went wrong.</div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
