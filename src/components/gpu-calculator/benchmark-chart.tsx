"use client";

import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
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

interface MultiFilterProps {
  label: string;
  selected: Set<string>;
  options: string[];
  onChange: (v: Set<string>) => void;
}

function MultiFilter({ label, selected, options, onChange }: MultiFilterProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const allSelected = selected.size === 0;

  const toggle = (val: string) => {
    const next = new Set(selected);
    if (next.has(val)) next.delete(val);
    else next.add(val);
    onChange(next);
  };

  const displayText = allSelected
    ? "All"
    : selected.size === 1
      ? [...selected][0]
      : `${selected.size} selected`;

  return (
    <div className="relative flex flex-col gap-1" ref={ref}>
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
      >
        <span className="truncate">{displayText}</span>
        <svg className="ml-1 h-3 w-3 shrink-0 opacity-50" viewBox="0 0 12 12"><path d="M3 5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
      </button>
      {open && (
        <div className="absolute top-full left-0 z-50 mt-1 max-h-60 min-w-[160px] overflow-y-auto rounded-md border border-border bg-background py-1 shadow-lg">
          <label className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted/50">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={() => onChange(new Set())}
              className="rounded"
            />
            All
          </label>
          {options.map((o) => (
            <label key={o} className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted/50">
              <input
                type="checkbox"
                checked={allSelected || selected.has(o)}
                onChange={() => {
                  if (allSelected) {
                    // switching from "All" to selecting everything except this one
                    const next = new Set(options);
                    next.delete(o);
                    onChange(next);
                  } else {
                    toggle(o);
                  }
                }}
                className="rounded"
              />
              {o}
            </label>
          ))}
        </div>
      )}
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

function parseParam(v: string | null): Set<string> {
  if (!v) return new Set();
  return new Set(v.split(",").filter(Boolean));
}

export function BenchmarkChart({ data }: { data: BenchmarkData }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const [gpuFilter, setGpuFilter] = useState<Set<string>>(() => parseParam(searchParams.get("gpu")));
  const [modelFilter, setModelFilter] = useState<Set<string>>(() => parseParam(searchParams.get("model")));
  const [engineFilter, setEngineFilter] = useState<Set<string>>(() => parseParam(searchParams.get("engine")));
  const [configFilter, setConfigFilter] = useState<Set<string>>(() => parseParam(searchParams.get("config")));
  const [logScale, setLogScale] = useState(() => searchParams.get("log") === "1");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Sync filter state to URL query params
  useEffect(() => {
    const params = new URLSearchParams();
    if (gpuFilter.size > 0) params.set("gpu", [...gpuFilter].join(","));
    if (modelFilter.size > 0) params.set("model", [...modelFilter].join(","));
    if (engineFilter.size > 0) params.set("engine", [...engineFilter].join(","));
    if (configFilter.size > 0) params.set("config", [...configFilter].join(","));
    if (logScale) params.set("log", "1");
    const qs = params.toString();
    router.replace(`${pathname}${qs ? `?${qs}` : ""}`, { scroll: false });
  }, [gpuFilter, modelFilter, engineFilter, configFilter, logScale, router, pathname]);

  const filteredPoints = useMemo(() => {
    return data.points.filter((p) => {
      if (gpuFilter.size > 0 && !gpuFilter.has(p.gpu)) return false;
      if (modelFilter.size > 0 && !modelFilter.has(p.model)) return false;
      if (engineFilter.size > 0 && !engineFilter.has(p.engine)) return false;
      if (configFilter.size > 0 && !configFilter.has(p.config)) return false;
      // Exclude zero values when log scale is on — they break log scale
      if (logScale && (p.throughputPerGpu <= 0 || p.e2eLatency <= 0)) return false;
      return true;
    });
  }, [data.points, gpuFilter, modelFilter, engineFilter, configFilter, logScale]);

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
        <MultiFilter label="GPU Type" selected={gpuFilter} options={data.gpus} onChange={setGpuFilter} />
        <MultiFilter label="Model" selected={modelFilter} options={data.models} onChange={setModelFilter} />
        <MultiFilter label="Engine" selected={engineFilter} options={data.engines} onChange={setEngineFilter} />
        <MultiFilter label="Parallelism" selected={configFilter} options={data.configs} onChange={setConfigFilter} />
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-muted-foreground">Log Scale</label>
          <button
            role="switch"
            aria-checked={logScale}
            onClick={() => setLogScale((v) => !v)}
            className={`relative inline-flex h-[28px] w-[48px] items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50 ${
              logScale ? "bg-primary" : "bg-border"
            }`}
          >
            <span
              className={`inline-block h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
                logScale ? "translate-x-[22px]" : "translate-x-[4px]"
              }`}
            />
          </button>
        </div>
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
              {...(logScale ? { scale: "log" as const, domain: ["dataMin", "dataMax"], allowDataOverflow: true } : {})}
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(Math.round(v))}
              label={{
                value: `Throughput/GPU (tok/s)${logScale ? " — log scale" : ""}`,
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
              {...(logScale ? { scale: "log" as const, domain: ["dataMin", "dataMax"], allowDataOverflow: true } : {})}
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => v >= 1 ? `${v.toFixed(0)}` : v.toFixed(2)}
              label={{
                value: `E2E Latency (s)${logScale ? " — log scale" : ""}`,
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
