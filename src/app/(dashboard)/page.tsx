"use client";

import { Suspense } from "react";
import { BenchmarkChart } from "@/components/gpu-calculator/benchmark-chart";
import { useBenchmarkData } from "./benchmark-data-provider";

function BenchmarkContent() {
  const data = useBenchmarkData();
  return <BenchmarkChart data={data} />;
}

export default function BenchmarkPage() {
  return (
    <Suspense fallback={<div className="p-4 text-sm text-muted-foreground">Loading…</div>}>
      <BenchmarkContent />
    </Suspense>
  );
}
