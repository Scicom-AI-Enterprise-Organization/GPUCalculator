import { readBenchmarkData } from "@/lib/read-benchmarks";
import { GpuCalculatorClient } from "./client";

export const metadata = {
  title: "GPU Calculator - Scicom AI",
  description: "GPU performance benchmark dashboard and VRAM estimator for LLM inference",
};

export default async function HomePage() {
  const data = await readBenchmarkData();

  return <GpuCalculatorClient data={data} />;
}
