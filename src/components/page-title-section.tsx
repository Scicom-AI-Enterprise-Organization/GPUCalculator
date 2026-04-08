"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { fadeIn, staggerContainer } from "@/lib/animations";

interface PageTitleSectionProps {
  /** Main title of the page */
  title: string;
  /** Description text below the title */
  description: string;
  /** Show back to home link (default: true) */
  showBackLink?: boolean;
  /** Custom back link href */
  backLinkHref?: string;
  /** Custom back link text */
  backLinkText?: string;
}

/**
 * Shared page title section with optional back navigation.
 * Used on sub-pages like Components, Documentation, Design System.
 */
export function PageTitleSection({
  title,
  description,
  showBackLink = true,
  backLinkHref = "/",
  backLinkText = "Back to Home",
}: PageTitleSectionProps) {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={staggerContainer}
      className="mb-12"
    >
      {showBackLink && (
        <motion.div variants={fadeIn} className="mb-4">
          <Link
            href={backLinkHref}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            {backLinkText}
          </Link>
        </motion.div>
      )}

      <motion.h1
        variants={fadeIn}
        className="mb-4 text-4xl font-bold tracking-tight"
      >
        {title}
      </motion.h1>

      <motion.p
        variants={fadeIn}
        className="max-w-2xl text-lg text-muted-foreground"
      >
        {description}
      </motion.p>
    </motion.div>
  );
}
