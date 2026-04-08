"use client";

import Image from "next/image";

interface PageFooterProps {
  /** Optional: Custom description text */
  description?: string;
}

/**
 * Shared footer component used across all pages.
 * Displays logo, Design Hub label, and credit text.
 */
export function PageFooter({
  description = "Built with Next.js, Tailwind CSS, and shadcn/ui.",
}: PageFooterProps) {
  return (
    <footer className="border-t border-border py-8">
      <div className="container mx-auto px-4 lg:px-8">
        <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
          <div className="flex items-center gap-2">
            <Image
              src="/images/scicom-logo.png"
              alt="Scicom"
              width={100}
              height={32}
              className="h-6 w-auto dark:brightness-0 dark:invert"
            />
            <span className="text-sm font-medium text-muted-foreground">
              Design Hub
            </span>
          </div>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </div>
    </footer>
  );
}
