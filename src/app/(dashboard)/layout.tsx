import { readBenchmarkData } from "@/lib/read-benchmarks";
import { GpuCalculatorShell } from "../client";
import { BenchmarkDataProvider } from "./benchmark-data-provider";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const data = await readBenchmarkData();

  return (
    <BenchmarkDataProvider data={data}>
      <GpuCalculatorShell>{children}</GpuCalculatorShell>
    </BenchmarkDataProvider>
  );
}
