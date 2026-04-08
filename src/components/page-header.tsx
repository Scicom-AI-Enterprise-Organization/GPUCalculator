"use client";

import Link from "next/link";
import Image from "next/image";
import { Sun, Moon } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BrandSwitcher } from "@/components/brand-switcher";
import { useBrand } from "@/components/brand-provider";
import { useThemeMode } from "@/hooks/use-theme-mode";

interface NavLink {
  href: string;
  label: string;
}

const navLinks: NavLink[] = [
  { href: "/design-system", label: "Design System" },
  { href: "/components", label: "Components" },
  { href: "/templates", label: "Templates" },
  { href: "/resources", label: "Resources" },
  { href: "/documentation", label: "Documentation" },
];

interface PageHeaderProps {
  /** Optional: Override the default navigation links */
  links?: NavLink[];
  /** Optional: Show/hide the Design Hub badge */
  showBadge?: boolean;
}

/**
 * Shared navigation header component used across all pages.
 * Includes logo, navigation links, brand switcher, and theme toggle.
 */
export function PageHeader({ links = navLinks, showBadge = true }: PageHeaderProps) {
  const { brand, setBrand } = useBrand();
  const { isDark, toggleTheme } = useThemeMode();

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="z-50 w-full shrink-0 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    >
      <nav className="flex h-16 items-center justify-between pl-3 pr-4 lg:pr-8">
        {/* Logo */}
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2">
            <Image
              src="/images/scicom-logo.png"
              alt="Scicom"
              width={120}
              height={40}
              className="h-8 w-auto dark:brightness-0 dark:invert"
            />
          </Link>

          {/* Version Badge */}
          {showBadge && (
            <div className="hidden items-center gap-2 md:flex">
              <Badge
                variant="outline"
                className="rounded-full px-3 py-1 text-xs font-medium"
              >
                Design Hub
              </Badge>
            </div>
          )}
        </div>

        {/* Navigation Links */}
        <div className="hidden items-center gap-8 md:flex">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center gap-3">
          <BrandSwitcher currentBrand={brand} onBrandChange={setBrand} />
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="rounded-full"
          >
            {isDark ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </Button>
        </div>
      </nav>
    </motion.header>
  );
}
