"use client";

import { PageHeader } from "@/components/page-header";
import { PageFooter } from "@/components/page-footer";
import { cn } from "@/lib/utils";

interface PageLayoutProps {
  children: React.ReactNode;
  showHeader?: boolean;
  showFooter?: boolean;
  mainClassName?: string;
  containerize?: boolean;
}

export function PageLayout({
  children,
  showHeader = true,
  showFooter = true,
  mainClassName,
  containerize = true,
}: PageLayoutProps) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {showHeader && <PageHeader />}

      <main
        className={cn(
          containerize && "container mx-auto px-4 py-8 lg:px-8",
          mainClassName
        )}
      >
        {children}
      </main>

      {showFooter && <PageFooter />}
    </div>
  );
}
