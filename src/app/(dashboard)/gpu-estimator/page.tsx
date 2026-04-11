"use client";

import { Suspense } from "react";
import { GpuEstimator } from "@/components/gpu-calculator/gpu-estimator";
import { useBenchmarkData } from "../benchmark-data-provider";

function GpuEstimatorInner() {
  const data = useBenchmarkData();
  return <GpuEstimator benchmarkData={data} />;
}

export default function GpuEstimatorPage() {
  return (
    <Suspense fallback={null}>
      <GpuEstimatorInner />
    </Suspense>
  );
}
