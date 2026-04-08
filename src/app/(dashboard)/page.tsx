"use client";

import { BenchmarkChart } from "@/components/gpu-calculator/benchmark-chart";
import { useBenchmarkData } from "./benchmark-data-provider";

export default function BenchmarkPage() {
  const data = useBenchmarkData();
  return <BenchmarkChart data={data} />;
}
