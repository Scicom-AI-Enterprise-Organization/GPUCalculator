"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronLeft, ChevronRight, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface NavItem {
  label: string;
  href: string;
  icon?: LucideIcon;
}

interface SidebarSection {
  title?: string;
  items: NavItem[];
}

interface PageSidebarProps {
  /** Navigation sections to display */
  sections: SidebarSection[];
  /** Toggle options (e.g., Product / Editorial) */
  toggleOptions?: {
    options: string[];
    defaultValue?: string;
    onChange?: (value: string) => void;
  };
  /** Current active toggle value */
  activeToggle?: string;
}

/**
 * Collapsible sidebar navigation component with icon support.
 * Collapsed state shows icons only. Expanded state shows icons + labels.
 */
export function PageSidebar({
  sections,
  toggleOptions,
  activeToggle,
}: PageSidebarProps) {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(true);
  const [toggle, setToggle] = useState(
    activeToggle || toggleOptions?.defaultValue || toggleOptions?.options[0] || ""
  );

  const handleToggle = (value: string) => {
    setToggle(value);
    toggleOptions?.onChange?.(value);
  };

  return (
    <div className="relative flex shrink-0">
      <aside
        className={cn(
          "overflow-y-auto border-r border-border transition-[width] duration-300 ease-in-out",
          isOpen ? "w-64" : "w-14"
        )}
      >
        <nav
          className="space-y-2 px-2 pt-8 pb-8"
        >
          {/* Toggle Switch — hidden when collapsed */}
          {isOpen && toggleOptions && toggleOptions.options.length > 0 && (
            <div className="mb-6">
              <div className="inline-flex rounded-full bg-muted p-1">
                {toggleOptions.options.map((option) => (
                  <button
                    key={option}
                    onClick={() => handleToggle(option)}
                    className={cn(
                      "rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
                      toggle === option
                        ? "bg-card text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Navigation Sections */}
          {sections.map((section, sectionIndex) => {
            if (section.title && isOpen) {
              // Collapsible titled section — only when expanded
              return (
                <Collapsible key={sectionIndex} defaultOpen={true}>
                  <CollapsibleTrigger className="flex w-full items-center justify-between rounded-md px-2 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground">
                    <span>{section.title}</span>
                    <ChevronRight className="h-3.5 w-3.5 transition-transform duration-200 [[data-state=open]>&]:rotate-90" />
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="space-y-0.5 pt-1">
                      {section.items.map((item) => {
                        const isActive = pathname === item.href;
                        const Icon = item.icon;
                        return (
                          <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                              "flex items-center rounded-md py-1.5 text-sm transition-colors",
                              isActive
                                ? "bg-primary/10 font-medium text-foreground"
                                : "text-muted-foreground hover:bg-muted hover:text-foreground"
                            )}
                          >
                            <span className="flex h-9 w-10 shrink-0 items-center justify-center">
                              {Icon && <Icon className="h-4 w-4" />}
                            </span>
                            <span>{item.label}</span>
                          </Link>
                        );
                      })}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              );
            }

            return (
              <div key={sectionIndex} className="space-y-0.5">
                {section.items.map((item) => {
                  const isActive = pathname === item.href;
                  const Icon = item.icon;

                  if (!isOpen) {
                    // Collapsed: icon-only with tooltip
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        title={item.label}
                        className={cn(
                          "flex h-9 w-10 items-center justify-center rounded-md transition-colors",
                          isActive
                            ? "bg-primary/10 text-foreground"
                            : "text-muted-foreground hover:bg-muted hover:text-foreground"
                        )}
                      >
                        {Icon ? (
                          <Icon className="h-4 w-4" />
                        ) : (
                          <span className="text-xs font-medium">
                            {item.label.charAt(0)}
                          </span>
                        )}
                      </Link>
                    );
                  }

                  // Expanded: icon + label
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={cn(
                        "flex items-center rounded-md py-1.5 text-sm transition-colors",
                        isActive
                          ? "bg-primary/10 font-medium text-foreground"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      )}
                    >
                      <span className="flex h-9 w-10 shrink-0 items-center justify-center">
                        {Icon && <Icon className="h-4 w-4" />}
                      </span>
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            );
          })}
        </nav>
      </aside>

      {/* Collapse/Expand toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="absolute -right-3 top-10 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm transition-colors hover:bg-muted hover:text-foreground"
        aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
      >
        {isOpen ? (
          <ChevronLeft className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
      </button>
    </div>
  );
}
