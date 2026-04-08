"use client";

import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import { Cpu } from "lucide-react";

export function PageHeader() {
  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="z-50 w-full shrink-0 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    >
      <nav className="flex h-14 items-center justify-between px-4 lg:px-8">
        <Link href="/" className="flex items-center gap-3">
          <Image
            src="/images/scicom-logo.png"
            alt="Scicom"
            width={100}
            height={32}
            className="h-7 w-auto dark:brightness-0 dark:invert"
          />
          <div className="hidden items-center gap-2 sm:flex">
            <span className="text-muted-foreground">/</span>
            <div className="flex items-center gap-1.5">
              <Cpu className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">GPU Calculator</span>
            </div>
          </div>
        </Link>

        <div className="flex items-center gap-4">
          <a
            href="https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            Benchmark Data
          </a>
        </div>
      </nav>
    </motion.header>
  );
}
