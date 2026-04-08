"use client";

import { createContext, useContext } from "react";
import type { BenchmarkData } from "@/lib/read-benchmarks";

const BenchmarkDataContext = createContext<BenchmarkData | null>(null);

export function useBenchmarkData(): BenchmarkData {
  const data = useContext(BenchmarkDataContext);
  if (!data) throw new Error("useBenchmarkData must be used within BenchmarkDataProvider");
  return data;
}

export function BenchmarkDataProvider({
  data,
  children,
}: {
  data: BenchmarkData;
  children: React.ReactNode;
}) {
  return (
    <BenchmarkDataContext.Provider value={data}>
      {children}
    </BenchmarkDataContext.Provider>
  );
}
