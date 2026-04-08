"use client";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Info, Check, AlertCircle, FileText, Loader, Inbox, CircleUser } from "lucide-react";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";

const feedbackSidebarSections = [
  {
    items: [
      { label: "Alerts", href: "#alerts", icon: AlertCircle },
      { label: "Loading States", href: "#loading", icon: Loader },
      { label: "Empty States", href: "#empty-states", icon: Inbox },
      { label: "Avatars", href: "#avatars", icon: CircleUser },
    ],
  },
];

export default function FeedbackPage() {
  return (
    <TwoColumnLayout
      title="Feedback"
      sidebarSections={feedbackSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Feedback components communicate status, progress, and information to users.
          Includes alerts, loading indicators, empty states, and avatars.
        </p>
      </div>

      {/* Alerts */}
      <section id="alerts" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Alerts</h2>
        <p className="mb-6 text-muted-foreground">
          Contextual messages for user feedback and notifications.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Info Alert" count={1}>
            <div className="w-full max-w-[220px]">
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  This is an informational message.
                </AlertDescription>
              </Alert>
            </div>
          </ComponentCard>

          <ComponentCard title="Success Alert" count={1}>
            <div className="w-full max-w-[220px] rounded-lg border border-success-200 bg-success-50 p-3 dark:border-success-800 dark:bg-success-900/30">
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-success-600" />
                <span className="text-sm text-success-700 dark:text-success-400">Operation completed successfully.</span>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Error Alert" count={1}>
            <div className="w-full max-w-[220px] rounded-lg border border-danger-200 bg-danger-50 p-3 dark:border-danger-800 dark:bg-danger-900/30">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-danger-600" />
                <span className="text-sm text-danger-700 dark:text-danger-400">Something went wrong.</span>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Loading States */}
      <section id="loading" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Loading States</h2>
        <p className="mb-6 text-muted-foreground">
          Visual indicators for asynchronous operations.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Spinner" count={1}>
            <div className="flex items-center gap-4">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            </div>
          </ComponentCard>

          <ComponentCard title="Skeleton" count={1}>
            <div className="space-y-2">
              <div className="h-4 w-32 animate-pulse rounded bg-muted" />
              <div className="h-4 w-24 animate-pulse rounded bg-muted" />
              <div className="h-4 w-28 animate-pulse rounded bg-muted" />
            </div>
          </ComponentCard>

          <ComponentCard title="Pulse Dots" count={1}>
            <div className="flex gap-1">
              <div className="h-2 w-2 animate-bounce rounded-full bg-primary" style={{ animationDelay: "0ms" }} />
              <div className="h-2 w-2 animate-bounce rounded-full bg-primary" style={{ animationDelay: "150ms" }} />
              <div className="h-2 w-2 animate-bounce rounded-full bg-primary" style={{ animationDelay: "300ms" }} />
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Empty States */}
      <section id="empty-states" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Empty States</h2>
        <p className="mb-6 text-muted-foreground">
          Placeholders when no content is available.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="No Items" count={1}>
            <div className="text-center p-4">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                <FileText className="h-6 w-6 text-muted-foreground" />
              </div>
              <div className="text-sm font-medium">No items found</div>
              <p className="mt-1 text-xs text-muted-foreground">
                Get started by creating your first item.
              </p>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Avatars */}
      <section id="avatars" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">Avatars</h2>
        <p className="mb-6 text-muted-foreground">
          Visual representation of users or entities.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Avatar Sizes" count={1}>
            <div className="flex items-end gap-2">
              <Avatar className="h-8 w-8">
                <AvatarFallback className="text-xs">SM</AvatarFallback>
              </Avatar>
              <Avatar className="h-10 w-10">
                <AvatarFallback className="text-sm">MD</AvatarFallback>
              </Avatar>
              <Avatar className="h-12 w-12">
                <AvatarFallback>LG</AvatarFallback>
              </Avatar>
            </div>
          </ComponentCard>

          <ComponentCard title="Avatar Stack" count={1}>
            <div className="flex items-center -space-x-3">
              <Avatar className="h-9 w-9 border-2 border-card">
                <AvatarFallback className="text-xs">A</AvatarFallback>
              </Avatar>
              <Avatar className="h-9 w-9 border-2 border-card">
                <AvatarFallback className="text-xs">B</AvatarFallback>
              </Avatar>
              <Avatar className="h-9 w-9 border-2 border-card">
                <AvatarFallback className="text-xs">C</AvatarFallback>
              </Avatar>
              <Avatar className="h-9 w-9 border-2 border-card">
                <AvatarFallback className="bg-primary text-xs text-white">+3</AvatarFallback>
              </Avatar>
            </div>
          </ComponentCard>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
