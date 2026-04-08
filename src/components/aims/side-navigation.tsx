"use client";

import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { LayoutDashboard, FolderOpen, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

interface SideNavItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  segment: string; // URL segment appended to base channel path
}

const sideNavItems: SideNavItem[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, segment: "" },
  { id: "cases", label: "Cases", icon: FolderOpen, segment: "/cases" },
  { id: "configuration", label: "Configuration", icon: Settings, segment: "/configuration" },
];

export function AimsSideNavigation() {
  const pathname = usePathname();
  const router = useRouter();

  // Extract the base channel path (e.g., /templates/aims/app/voice-agent)
  const segments = pathname.split("/");
  // /templates/aims/app/[channel] = index 0-4
  const basePath = segments.slice(0, 5).join("/");

  // Determine active item
  const subPath = segments.slice(5).join("/");
  const activeItem = sideNavItems.find((item) => {
    if (item.segment === "") return subPath === "";
    return subPath.startsWith(item.segment.replace("/", ""));
  })?.id || "overview";

  return (
    <aside className="flex w-56 flex-col border-r border-border bg-card">
      {/* Channel Title */}
      <div className="px-4 py-4 border-b border-border">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Navigation
        </h2>
      </div>

      {/* Nav Items */}
      <nav className="flex flex-col gap-1 p-3">
        {sideNavItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeItem === item.id;
          const href = `${basePath}${item.segment}`;

          return (
            <Button
              key={item.id}
              variant={isActive ? "secondary" : "ghost"}
              size="sm"
              onClick={() => router.push(href)}
              className={cn(
                "w-full justify-start gap-3 px-3",
                isActive
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Button>
          );
        })}
      </nav>
    </aside>
  );
}
