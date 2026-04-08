"use client";

import { GpuEstimator } from "@/components/gpu-calculator/gpu-estimator";
import { useBenchmarkData } from "../benchmark-data-provider";

export default function GpuEstimatorPage() {
  const data = useBenchmarkData();
  return <GpuEstimator benchmarkData={data} />;
}
