"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Cpu, BarChart3 } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { PageFooter } from "@/components/page-footer";

const tabs = [
  { id: "/", label: "Benchmark Results", icon: BarChart3 },
  { id: "/gpu-estimator", label: "GPU Estimator", icon: Cpu },
] as const;

export function GpuCalculatorShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <PageHeader />

      <main className="flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          {/* Title */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold tracking-tight">GPU Performance Calculator</h1>
            <p className="mt-2 text-muted-foreground">
              Explore LLM inference benchmarks across GPU types and estimate hardware requirements for your model.
            </p>
          </div>

          {/* Tabs */}
          <div className="mb-6 flex gap-0 rounded-lg border border-border bg-muted/30 p-1">
            {tabs.map((tab, i) => {
              const Icon = tab.icon;
              const isActive = pathname === tab.id;
              return (
                <Link
                  key={tab.id}
                  href={tab.id}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2.5 text-sm font-medium transition-colors ${
                    i > 0 ? "border-l border-border" : ""
                  } ${
                    isActive
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </Link>
              );
            })}
          </div>

          {/* Content */}
          {children}
        </div>
      </main>

      <PageFooter />
    </div>
  );
}
