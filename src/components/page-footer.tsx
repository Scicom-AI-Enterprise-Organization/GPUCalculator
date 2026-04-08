"use client";

import Image from "next/image";

export function PageFooter() {
  return (
    <footer className="border-t border-border bg-muted/30">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <Image
              src="/images/scicom-logo.png"
              alt="Scicom"
              width={100}
              height={32}
              className="h-6 w-auto dark:brightness-0 dark:invert"
            />
            <span className="text-xs text-muted-foreground">
              AI Enterprise Organization
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
            <a
              href="https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq"
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:text-foreground"
            >
              Benchmark Repository
            </a>
            <span className="hidden sm:inline">&middot;</span>
            <span>Built with Next.js &amp; Recharts</span>
            <span className="hidden sm:inline">&middot;</span>
            <span>Data: vLLM &amp; SGLang benchmarks across B200, H200, H100, A100</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
