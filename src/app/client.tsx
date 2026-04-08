"use client";

import { useState } from "react";
import { Cpu, BarChart3 } from "lucide-react";
import { BenchmarkChart } from "@/components/gpu-calculator/benchmark-chart";
import { GpuEstimator } from "@/components/gpu-calculator/gpu-estimator";
import type { BenchmarkData } from "@/lib/read-benchmarks";
import { PageHeader } from "@/components/page-header";
import { PageFooter } from "@/components/page-footer";

const tabs = [
  { id: "benchmark", label: "Benchmark Results", icon: BarChart3 },
  { id: "estimator", label: "GPU Estimator", icon: Cpu },
] as const;

type Tab = (typeof tabs)[number]["id"];

export function GpuCalculatorClient({ data }: { data: BenchmarkData }) {
  const [activeTab, setActiveTab] = useState<Tab>("benchmark");

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
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2.5 text-sm font-medium transition-colors ${
                    i > 0 ? "border-l border-border" : ""
                  } ${
                    activeTab === tab.id
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Content */}
          {activeTab === "benchmark" ? (
            <BenchmarkChart data={data} />
          ) : (
            <GpuEstimator />
          )}
        </div>
      </main>

      <PageFooter />
    </div>
  );
}
