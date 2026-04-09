"use client";

import { useMemo, useState, useEffect } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ZAxis,
  Cell,
  LabelList,
} from "recharts";
import type { BenchmarkData, BenchmarkPoint } from "@/lib/read-benchmarks";

const MODEL_COLORS: Record<string, string> = {
  "gpt-oss-120b": "#f97316",
  "qwen3-32b": "#22c55e",
  "qwen3-14b": "#16a34a",
  "qwen3-8b": "#4ade80",
  "qwen3.5-122b": "#059669",
  "qwen3.5-35b": "#34d399",
  "qwen3.5-27b": "#6ee7b7",
  "glm-4.7": "#0891b2",
  "glm-4.7-fp8": "#8b5cf6",
  "llama3.1-70b": "#ef4444",
};

const GPU_SHAPES: Record<string, string> = {
  "B200": "circle",
  "H200 SXM": "diamond",
  "H100 SXM": "square",
  "A100 SXM": "triangle",
};

function getSeriesKey(p: BenchmarkPoint) {
  return `${p.engine}_${p.gpu}_${p.model}_${p.config}`;
}

function getColor(model: string) {
  return MODEL_COLORS[model] || "#6b7280";
}

interface FilterSelectProps {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}

function FilterSelect({ label, value, options, onChange }: FilterSelectProps) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
      >
        <option value="All">All</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: BenchmarkPoint & { seriesKey: string };
  }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-background/95 p-3 shadow-lg backdrop-blur">
      <div className="mb-2 font-semibold text-foreground">
        {d.model} on {d.gpu}
      </div>
      <div className="space-y-1 text-xs text-muted-foreground">
        <div>Engine: <span className="text-foreground">{d.engine}</span></div>
        <div>Config: <span className="text-foreground">{d.config}</span></div>
        <div>Context: <span className="text-foreground">{d.ctx.toLocaleString()} tokens</span></div>
        <div>Throughput/GPU: <span className="text-foreground">{d.throughputPerGpu.toLocaleString()} tok/s</span></div>
        <div>E2E Latency: <span className="text-foreground">{d.e2eLatency.toFixed(2)}s</span></div>
        <div>TTFT: <span className="text-foreground">{d.ttft.toFixed(0)}ms</span></div>
      </div>
    </div>
  );
}

export function BenchmarkChart({ data }: { data: BenchmarkData }) {
  const [gpuFilter, setGpuFilter] = useState("All");
  const [modelFilter, setModelFilter] = useState("All");
  const [engineFilter, setEngineFilter] = useState("All");
  const [configFilter, setConfigFilter] = useState("All");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const filteredPoints = useMemo(() => {
    return data.points.filter((p) => {
      if (gpuFilter !== "All" && p.gpu !== gpuFilter) return false;
      if (modelFilter !== "All" && p.model !== modelFilter) return false;
      if (engineFilter !== "All" && p.engine !== engineFilter) return false;
      if (configFilter !== "All" && p.config !== configFilter) return false;
      return true;
    });
  }, [data.points, gpuFilter, modelFilter, engineFilter, configFilter]);

  const series = useMemo(() => {
    const grouped = new Map<string, (BenchmarkPoint & { seriesKey: string })[]>();
    for (const p of filteredPoints) {
      const key = getSeriesKey(p);
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push({ ...p, seriesKey: key });
    }
    // Sort each series by context length for proper line connection
    for (const [, pts] of grouped) {
      pts.sort((a, b) => a.ctx - b.ctx);
    }
    return grouped;
  }, [filteredPoints]);

  const seriesEntries = useMemo(() => [...series.entries()], [series]);

  return (
    <div className="min-w-0">
      {/* Filters */}
      <div className="mb-6 flex flex-wrap gap-4">
        <FilterSelect label="GPU Type" value={gpuFilter} options={data.gpus} onChange={setGpuFilter} />
        <FilterSelect label="Model" value={modelFilter} options={data.models} onChange={setModelFilter} />
        <FilterSelect label="Engine" value={engineFilter} options={data.engines} onChange={setEngineFilter} />
        <FilterSelect label="Parallelism" value={configFilter} options={data.configs} onChange={setConfigFilter} />
      </div>

      {/* Chart */}
      <div className="rounded-xl border border-border bg-card p-2 sm:p-4">
        <h3 className="mb-1 text-sm font-semibold sm:text-base">Token Throughput per GPU vs End-to-End Latency</h3>
        <p className="mb-4 text-xs text-muted-foreground">
          8 GPUs &middot; 100 concurrent requests &middot; 128 output tokens &middot; Points labeled with context length
        </p>
        <div className="h-[350px] w-full min-w-0 sm:h-[500px]">
        {mounted ? (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis
              type="number"
              dataKey="throughputPerGpu"
              name="Throughput/GPU"
              unit=" tok/s"
              tick={{ fontSize: 10 }}
              label={{
                value: "Throughput/GPU (tok/s)",
                position: "bottom",
                offset: 0,
                style: { fontSize: 10 },
              }}
            />
            <YAxis
              type="number"
              dataKey="e2eLatency"
              name="E2E Latency"
              unit="s"
              tick={{ fontSize: 10 }}
              label={{
                value: "E2E Latency (s)",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                style: { fontSize: 10 },
              }}
              width={50}
            />
            <ZAxis range={[20, 20]} />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 10, paddingTop: 8, lineHeight: "1.6" }}
            />
            {seriesEntries.map(([key, pts]) => {
              const sample = pts[0];
              const color = getColor(sample.model);
              const label = `${sample.model} (${sample.engine}, ${sample.config})`;
              return (
                <Scatter
                  key={key}
                  name={label}
                  data={pts}
                  fill={color}
                  stroke={color}
                  strokeWidth={2}
                  line={{ strokeDasharray: sample.engine === "SGLang" ? "6 3" : undefined }}
                  legendType={GPU_SHAPES[sample.gpu] as "circle" | "diamond" | "square" | "triangle" || "circle"}
                >
                  <LabelList
                    dataKey="ctx"
                    position="top"
                    style={{ fontSize: 9, fill: color }}
                    formatter={((v: unknown) => {
                      const n = Number(v);
                      if (isNaN(n)) return String(v ?? "");
                      if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}K`;
                      return String(n);
                    }) as (label: unknown) => string}
                  />
                </Scatter>
              );
            })}
          </ScatterChart>
        </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Loading chart…
          </div>
        )}
        </div>
      </div>

      {/* Summary stats */}
      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-xs text-muted-foreground">Total Benchmarks</div>
          <div className="text-xl font-bold">{filteredPoints.length}</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-xs text-muted-foreground">Configurations</div>
          <div className="text-xl font-bold">{series.size}</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-xs text-muted-foreground">Max Throughput/GPU</div>
          <div className="text-xl font-bold">
            {filteredPoints.length > 0
              ? Math.max(...filteredPoints.map((p) => p.throughputPerGpu)).toLocaleString()
              : 0}{" "}
            <span className="text-xs font-normal text-muted-foreground">tok/s</span>
          </div>
        </div>
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-xs text-muted-foreground">Min Latency</div>
          <div className="text-xl font-bold">
            {filteredPoints.length > 0
              ? Math.min(...filteredPoints.map((p) => p.e2eLatency)).toFixed(2)
              : 0}{" "}
            <span className="text-xs font-normal text-muted-foreground">s</span>
          </div>
        </div>
      </div>
    </div>
  );
}
