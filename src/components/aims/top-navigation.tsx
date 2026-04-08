"use client";

import { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useThemeMode } from "@/hooks/use-theme-mode";
import {
  Phone,
  Bot,
  Mail,
  Share2,
  Sun,
  Moon,
  User,
  Settings,
  LogOut,
  ChevronDown,
  Circle,
  Coffee,
  Bell,
} from "lucide-react";
import { cn } from "@/lib/utils";

type UserStatus = "online" | "offline" | "break";

interface Channel {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  href: string;
}

const channels: Channel[] = [
  { id: "voice-agent", label: "Voice Agent", icon: Phone, href: "/templates/aims/app/voice-agent" },
  { id: "chatbot", label: "Chatbot", icon: Bot, href: "/templates/aims/app/chatbot" },
  { id: "email", label: "Email", icon: Mail, href: "/templates/aims/app/email" },
  { id: "social-media", label: "Social Media", icon: Share2, href: "/templates/aims/app/social-media" },
];

const statusConfig: Record<UserStatus, { label: string; color: string; icon: React.ComponentType<{ className?: string }> }> = {
  online: { label: "Online", color: "bg-success-500", icon: Circle },
  offline: { label: "Offline", color: "bg-muted-foreground", icon: Circle },
  break: { label: "Break", color: "bg-warning-500", icon: Coffee },
};

export function AimsTopNavigation() {
  const pathname = usePathname();
  const router = useRouter();
  const { isDark, toggleTheme } = useThemeMode();
  const [userStatus, setUserStatus] = useState<UserStatus>("online");

  const activeChannel = channels.find((c) => pathname.startsWith(c.href))?.id || "voice-agent";
  const currentStatus = statusConfig[userStatus];

  return (
    <header className="flex h-14 items-center border-b border-border bg-card px-4">
      {/* Left — Logo / Brand */}
      <div className="flex items-center gap-3 mr-8">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
          <span className="text-sm font-bold text-primary-foreground">A</span>
        </div>
        <span className="text-sm font-semibold text-foreground font-display tracking-tight">AIMS</span>
      </div>

      {/* Center — Channel Tabs */}
      <nav className="flex flex-1 items-center justify-center gap-1">
        {channels.map((channel) => {
          const Icon = channel.icon;
          const isActive = activeChannel === channel.id;
          return (
            <Button
              key={channel.id}
              variant={isActive ? "default" : "ghost"}
              size="sm"
              onClick={() => router.push(channel.href)}
              className={cn(
                "gap-2",
                !isActive && "text-muted-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {channel.label}
            </Button>
          );
        })}
      </nav>

      {/* Right — Status, Notifications, Theme, Profile */}
      <div className="flex items-center gap-2">
        {/* User Status Selector */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
              <span className={cn("h-2 w-2 rounded-full", currentStatus.color)} />
              <span className="text-xs">{currentStatus.label}</span>
              <ChevronDown className="h-3 w-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-40">
            <DropdownMenuLabel className="text-xs text-muted-foreground">Set Status</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => setUserStatus("online")}>
              <Circle className="h-3 w-3 fill-success-500 text-success-500" />
              Online
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setUserStatus("break")}>
              <Coffee className="h-3 w-3 text-warning-500" />
              Break
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setUserStatus("offline")}>
              <Circle className="h-3 w-3 fill-muted-foreground text-muted-foreground" />
              Offline
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Notifications */}
        <Button variant="ghost" size="icon-sm" className="relative">
          <Bell className="h-4 w-4" />
          <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-[10px] font-medium text-white">
            3
          </span>
        </Button>

        {/* Theme Toggle */}
        <Button variant="ghost" size="icon-sm" onClick={toggleTheme}>
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        {/* Profile Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="gap-2 pl-2 pr-1">
              <Avatar className="h-7 w-7">
                <AvatarFallback className="bg-primary/10 text-primary text-xs">NA</AvatarFallback>
              </Avatar>
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>
              <div className="flex flex-col">
                <span className="text-sm font-medium">Nash Ahmad</span>
                <span className="text-xs text-muted-foreground">nash@scicom.com.my</span>
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuItem>
                <User className="h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="h-4 w-4" />
                Settings
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive">
              <LogOut className="h-4 w-4" />
              Sign out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
