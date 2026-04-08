"use client";

import * as React from "react";
import { useState } from "react";
import {
  SlidersHorizontal,
  Search,
  User,
  Clock,
  Smile,
  Languages,
  LayoutGrid,
  List,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";

// Types
interface FilterOption {
  label: string;
  enabled: boolean;
}

interface FilterSection {
  id: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  options: FilterOption[];
  expanded: boolean;
}

interface AIMSFilterProps {
  className?: string;
  onFilterChange?: (filters: Record<string, FilterOption[]>) => void;
}

// Standard tab sections
const standardSections: FilterSection[] = [
  {
    id: "users",
    title: "Users",
    icon: User,
    options: [{ label: "VIP", enabled: false }],
    expanded: true,
  },
  {
    id: "time",
    title: "Time",
    icon: Clock,
    options: [
      { label: "< 1 Minute", enabled: false },
      { label: "< 2 Minute", enabled: false },
      { label: "< 3 Minute", enabled: false },
    ],
    expanded: true,
  },
  {
    id: "sentiment",
    title: "Sentiment",
    icon: Smile,
    options: [
      { label: "Positive", enabled: false },
      { label: "Negative", enabled: false },
      { label: "Neutral", enabled: false },
    ],
    expanded: true,
  },
  {
    id: "language",
    title: "Language",
    icon: Languages,
    options: [{ label: "English", enabled: false }],
    expanded: true,
  },
];

// Advanced tab sections
const advancedSections: FilterSection[] = [
  {
    id: "users",
    title: "Users",
    icon: User,
    options: [{ label: "VIP", enabled: false }],
    expanded: true,
  },
  {
    id: "valueStream",
    title: "Value Stream",
    icon: Clock,
    options: [
      { label: "Assurance", enabled: false },
      { label: "Billing", enabled: false },
      { label: "Fulfillment", enabled: false },
    ],
    expanded: true,
  },
  {
    id: "categoryType",
    title: "Category Type",
    icon: Smile,
    options: [
      { label: "Enquiry", enabled: false },
      { label: "Negative", enabled: false },
      { label: "Neutral", enabled: false },
    ],
    expanded: true,
  },
];

export function AIMSFilter({ className, onFilterChange }: AIMSFilterProps) {
  const [activeTab, setActiveTab] = useState<"standard" | "advanced">("standard");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [globalSearch, setGlobalSearch] = useState("");
  const [sections, setSections] = useState<FilterSection[]>(standardSections);

  // Handle tab change
  const handleTabChange = (tab: "standard" | "advanced") => {
    setActiveTab(tab);
    setSections(tab === "standard" ? standardSections : advancedSections);
  };

  // Toggle section expand/collapse
  const toggleSection = (sectionId: string) => {
    setSections((prev) =>
      prev.map((section) =>
        section.id === sectionId
          ? { ...section, expanded: !section.expanded }
          : section
      )
    );
  };

  // Toggle option
  const toggleOption = (sectionId: string, optionIndex: number) => {
    setSections((prev) =>
      prev.map((section) =>
        section.id === sectionId
          ? {
              ...section,
              options: section.options.map((opt, idx) =>
                idx === optionIndex ? { ...opt, enabled: !opt.enabled } : opt
              ),
            }
          : section
      )
    );
  };

  return (
    <div className={cn("w-full max-w-md bg-card", className)}>
      {/* Header */}
      <div className="flex items-center gap-2 p-4">
        <SlidersHorizontal className="h-5 w-5" />
        <span className="text-base font-semibold">Filter</span>
      </div>

      {/* Tab Switcher */}
      <div className="mx-4 mb-4">
        <div className="flex rounded-full border border-border bg-muted p-1">
          <button
            onClick={() => handleTabChange("standard")}
            className={cn(
              "flex-1 rounded-full px-6 py-2 text-sm font-medium transition-all",
              activeTab === "standard"
                ? "bg-card shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Standard
          </button>
          <button
            onClick={() => handleTabChange("advanced")}
            className={cn(
              "flex-1 rounded-full px-6 py-2 text-sm font-medium transition-all",
              activeTab === "advanced"
                ? "bg-card shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Advanced
          </button>
        </div>
      </div>

      {/* Session Views */}
      <div className="flex items-center justify-between border-b border-border px-4 pb-4">
        <span className="text-sm font-medium">Session Views</span>
        <div className="flex rounded-lg border border-border">
          <button
            onClick={() => setViewMode("grid")}
            className={cn(
              "p-2 transition-colors",
              viewMode === "grid"
                ? "bg-muted"
                : "hover:bg-muted"
            )}
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={cn(
              "p-2 transition-colors",
              viewMode === "list"
                ? "bg-muted"
                : "hover:bg-muted"
            )}
          >
            <List className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Global Search - Only on Standard tab */}
      {activeTab === "standard" && (
        <div className="border-b border-border p-4">
          <div className="mb-3 flex items-center gap-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Global Search</span>
          </div>
          <Textarea
            placeholder="Session ID, Room SID, Username, Room Name, Content Type."
            value={globalSearch}
            onChange={(e) => setGlobalSearch(e.target.value)}
            className="min-h-[100px] resize-none text-sm"
            maxLength={140}
          />
          <p className="mt-2 text-xs text-muted-foreground">
            {globalSearch.length}/140 characters limit
          </p>
        </div>
      )}

      {/* Filter Sections */}
      <div className="divide-y divide-border">
        {sections.map((section) => (
          <div key={section.id} className="p-4">
            {/* Section Header */}
            <button
              onClick={() => toggleSection(section.id)}
              className="flex w-full items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <section.icon className="h-5 w-5 text-muted-foreground" />
                <span className="text-sm font-semibold">{section.title}</span>
              </div>
              {section.expanded ? (
                <ChevronUp className="h-5 w-5 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
              )}
            </button>

            {/* Section Options */}
            {section.expanded && (
              <div className="mt-3 space-y-3 pl-8">
                {section.options.map((option, idx) => (
                  <div
                    key={option.label}
                    className="flex items-center justify-between"
                  >
                    <span className="text-sm text-foreground">
                      {option.label}
                    </span>
                    <Switch
                      checked={option.enabled}
                      onCheckedChange={() => toggleOption(section.id, idx)}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
