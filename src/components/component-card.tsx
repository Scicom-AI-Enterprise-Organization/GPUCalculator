"use client";

import { motion } from "framer-motion";
import { fadeIn } from "@/lib/animations";
import { cn } from "@/lib/utils";
import Link from "next/link";

interface ComponentCardProps {
  /** Card title */
  title: string;
  /** Number of components in this category */
  count: number;
  /** Preview content to display */
  children: React.ReactNode;
  /** Optional: Click handler */
  onClick?: () => void;
  /** Optional: Link to detail page */
  href?: string;
  /** Optional: Additional CSS classes */
  className?: string;
}

/**
 * Preview card for component categories.
 * Shows a preview area with hover effects and component count.
 */
export function ComponentCard({
  title,
  count,
  children,
  onClick,
  href,
  className,
}: ComponentCardProps) {
  const content = (
    <motion.div
      variants={fadeIn}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
      className={cn("group cursor-pointer", className)}
      onClick={onClick}
    >
      <div className="rounded-xl border border-border bg-card transition-all duration-200 group-hover:border-primary group-hover:shadow-lg">
        {/* Preview Area */}
        <div className="flex min-h-[200px] items-center justify-center rounded-t-xl bg-muted/50 p-8">
          <div className="flex items-center justify-center">
            {children}
          </div>
        </div>
        {/* Card Info */}
        <div className="border-t border-border p-4">
          <h3 className="font-semibold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">{count} components</p>
        </div>
      </div>
    </motion.div>
  );

  if (href) {
    return <Link href={href}>{content}</Link>;
  }

  return content;
}
