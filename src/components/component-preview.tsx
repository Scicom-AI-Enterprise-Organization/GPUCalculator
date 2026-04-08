"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  Sun,
  Moon,
  Copy,
  Check,
  Settings,
  ChevronDown,
  Code2,
  Eye,
} from "lucide-react";

interface ComponentPreviewProps {
  /** The component(s) to preview */
  children: React.ReactNode;
  /** Optional title for the preview */
  title?: string;
  /** Code example to display in code view */
  code?: string;
  /** Minimum height for the preview area */
  minHeight?: string;
  /** Additional CSS classes */
  className?: string;
}

type ThemeOption = "default" | "scicom";

/**
 * Component preview container with toolbar for design system documentation.
 * Features Preview/Code toggle, theme selection, dark mode toggle, and copy functionality.
 */
export function ComponentPreview({
  children,
  title,
  code,
  minHeight = "300px",
  className,
}: ComponentPreviewProps) {
  const [activeTab, setActiveTab] = useState<"preview" | "code">("preview");
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [theme, setTheme] = useState<ThemeOption>("default");
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (code) {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const themeLabels: Record<ThemeOption, string> = {
    default: "Default",
    scicom: "Scicom",
  };

  return (
    <div className={cn("overflow-hidden rounded-xl border border-border", className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-border bg-muted/50 px-4 py-2">
        {/* Left side - Preview/Code toggle */}
        <div className="flex items-center gap-2">
          {title && (
            <span className="mr-4 text-sm font-medium text-foreground">{title}</span>
          )}
          <Tabs
            value={activeTab}
            onValueChange={(value) => setActiveTab(value as "preview" | "code")}
          >
            <TabsList className="h-8">
              <TabsTrigger value="preview" className="h-7 gap-1.5 px-3 text-xs">
                <Eye className="h-3.5 w-3.5" />
                Preview
              </TabsTrigger>
              <TabsTrigger
                value="code"
                className="h-7 gap-1.5 px-3 text-xs"
                disabled={!code}
              >
                <Code2 className="h-3.5 w-3.5" />
                Code
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {/* Right side - Theme dropdown, dark mode toggle, copy, settings */}
        <div className="flex items-center gap-2">
          {/* Theme Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 gap-1 px-2 text-xs">
                Theme: {themeLabels[theme]}
                <ChevronDown className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setTheme("default")}>
                Default
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("scicom")}>
                Scicom
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Dark Mode Toggle */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setIsDarkMode(!isDarkMode)}
            title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDarkMode ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </Button>

          {/* Copy Button */}
          <Button
            variant="ghost"
            size="sm"
            className="h-8 gap-1.5 px-2 text-xs"
            onClick={handleCopy}
            disabled={!code}
            title="Copy code"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-success-500" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                Copy
              </>
            )}
          </Button>

          {/* Settings Button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            title="Settings"
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Preview Area */}
      {activeTab === "preview" && (
        <div
          className={cn(
            "flex items-center justify-center p-12 transition-colors duration-200 md:p-16",
            isDarkMode
              ? "bg-background"
              : "bg-muted/50"
          )}
          style={{ minHeight }}
        >
          <div
            className={cn(
              "flex items-center justify-center",
              isDarkMode && "dark"
            )}
          >
            {children}
          </div>
        </div>
      )}

      {/* Code View */}
      {activeTab === "code" && code && (
        <div
          className="overflow-auto bg-background p-6"
          style={{ minHeight }}
        >
          <pre className="text-sm text-foreground">
            <code>{code}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

/**
 * Simplified version for quick showcases without the full toolbar.
 * Just shows the component on a clean background.
 */
export function SimplePreview({
  children,
  className,
  minHeight = "200px",
}: {
  children: React.ReactNode;
  className?: string;
  minHeight?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-center rounded-xl border border-border bg-muted/50 p-8",
        className
      )}
      style={{ minHeight }}
    >
      <div className="flex items-center justify-center">{children}</div>
    </div>
  );
}
