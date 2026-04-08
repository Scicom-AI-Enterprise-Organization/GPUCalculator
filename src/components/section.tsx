"use client";

import { motion } from "framer-motion";
import { staggerContainer, fadeIn, defaultViewport } from "@/lib/animations";
import { cn } from "@/lib/utils";

interface SectionProps {
  /** Section title */
  title?: string;
  /** Optional description below title */
  description?: string;
  /** Section content */
  children: React.ReactNode;
  /** Additional CSS classes for the section */
  className?: string;
  /** Additional CSS classes for the content container */
  contentClassName?: string;
  /** Whether to animate on scroll into view (default: true) */
  animate?: boolean;
  /** HTML id for anchor linking */
  id?: string;
}

/**
 * Shared section wrapper with consistent styling and animations.
 * Use for content sections on any page.
 */
export function Section({
  title,
  description,
  children,
  className,
  contentClassName,
  animate = true,
  id,
}: SectionProps) {
  const Wrapper = animate ? motion.section : "section";
  const TitleWrapper = animate ? motion.h2 : "h2";
  const DescWrapper = animate ? motion.p : "p";

  const wrapperProps = animate
    ? {
        initial: "hidden",
        whileInView: "visible",
        viewport: defaultViewport,
        variants: staggerContainer,
      }
    : {};

  const titleProps = animate ? { variants: fadeIn } : {};

  return (
    <Wrapper
      id={id}
      className={cn("mb-12", className)}
      {...wrapperProps}
    >
      {title && (
        <TitleWrapper
          className="mb-2 text-2xl font-bold"
          {...titleProps}
        >
          {title}
        </TitleWrapper>
      )}

      {description && (
        <DescWrapper
          className="mb-6 text-muted-foreground"
          {...(animate ? { variants: fadeIn } : {})}
        >
          {description}
        </DescWrapper>
      )}

      <div className={contentClassName}>{children}</div>
    </Wrapper>
  );
}
