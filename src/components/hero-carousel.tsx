"use client";

import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Check,
  DollarSign,
  Users,
  TrendingUp,
  Settings,
  Home,
  AlertCircle,
  Layers,
  FileText,
  X,
  Plus,
  Search,
} from "lucide-react";

// Mini component previews for the carousel - LARGER sizes
const componentPreviews = [
  // Charts
  {
    id: "chart-1",
    content: (
      <div className="w-64 rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Revenue</span>
          <Badge variant="secondary" className="text-xs">+12%</Badge>
        </div>
        <div className="flex h-20 items-end gap-1.5">
          {[40, 65, 45, 80, 55, 90, 70].map((h, i) => (
            <div key={i} className="flex-1 rounded-t bg-primary/80" style={{ height: `${h}%` }} />
          ))}
        </div>
      </div>
    ),
  },
  // Stats Card
  {
    id: "stats-1",
    content: (
      <div className="flex gap-3">
        <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
          <DollarSign className="mb-2 h-5 w-5 text-primary" />
          <div className="text-2xl font-bold">$45.2k</div>
          <div className="text-xs text-muted-foreground">Revenue</div>
        </div>
        <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
          <Users className="mb-2 h-5 w-5 text-green-500" />
          <div className="text-2xl font-bold">2,340</div>
          <div className="text-xs text-muted-foreground">Users</div>
        </div>
      </div>
    ),
  },
  // Progress
  {
    id: "progress-1",
    content: (
      <div className="w-56 space-y-3 rounded-xl border border-border bg-card p-4 shadow-sm">
        <div>
          <div className="mb-2 flex justify-between text-xs">
            <span>Sales</span><span>78%</span>
          </div>
          <div className="h-2.5 w-full rounded-full bg-muted">
            <div className="h-2.5 w-[78%] rounded-full bg-primary" />
          </div>
        </div>
        <div>
          <div className="mb-2 flex justify-between text-xs">
            <span>Growth</span><span>45%</span>
          </div>
          <div className="h-2.5 w-full rounded-full bg-muted">
            <div className="h-2.5 w-[45%] rounded-full bg-green-500" />
          </div>
        </div>
      </div>
    ),
  },
  // KPI Widget
  {
    id: "kpi-1",
    content: (
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between gap-6">
          <div>
            <div className="text-xs text-muted-foreground">Conversion</div>
            <div className="text-2xl font-bold">3.24%</div>
          </div>
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
            <TrendingUp className="h-6 w-6 text-green-600" />
          </div>
        </div>
        <div className="mt-2 text-xs text-green-600">+0.5% from last week</div>
      </div>
    ),
  },
  // Profile Card
  {
    id: "profile-1",
    content: (
      <div className="rounded-xl border border-border bg-card p-5 text-center shadow-sm">
        <Avatar className="mx-auto mb-3 h-14 w-14">
          <AvatarFallback className="text-lg">JD</AvatarFallback>
        </Avatar>
        <div className="text-base font-medium">John Doe</div>
        <div className="text-xs text-muted-foreground">Product Designer</div>
      </div>
    ),
  },
  // Pricing Card
  {
    id: "pricing-1",
    content: (
      <div className="rounded-xl border-2 border-primary bg-card p-4 shadow-sm">
        <Badge className="mb-3 text-xs">Popular</Badge>
        <div className="text-2xl font-bold">$29<span className="text-base font-normal text-muted-foreground">/mo</span></div>
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Check className="h-4 w-4 text-green-500" /><span>10 Projects</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Check className="h-4 w-4 text-green-500" /><span>Unlimited Users</span>
          </div>
        </div>
      </div>
    ),
  },
  // Feature Card
  {
    id: "feature-1",
    content: (
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
          <Layers className="h-5 w-5 text-primary" />
        </div>
        <div className="text-base font-medium">Feature Title</div>
        <div className="text-sm text-muted-foreground">Brief description here.</div>
      </div>
    ),
  },
  // Modal Preview
  {
    id: "modal-1",
    content: (
      <div className="w-56 rounded-xl border border-border bg-card shadow-lg">
        <div className="flex items-center justify-between border-b border-border p-3">
          <span className="text-sm font-medium">Dialog Title</span>
          <X className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="p-3">
          <p className="text-sm text-muted-foreground">Dialog content here...</p>
        </div>
        <div className="flex justify-end gap-2 border-t border-border p-3">
          <Button size="sm" variant="outline" className="h-8 text-xs">Cancel</Button>
          <Button size="sm" className="h-8 text-xs">Confirm</Button>
        </div>
      </div>
    ),
  },
  // Toast
  {
    id: "toast-1",
    content: (
      <div className="space-y-2">
        <div className="flex items-center gap-3 rounded-xl border border-border bg-card p-3 shadow-sm">
          <Check className="h-5 w-5 text-green-500" />
          <span className="text-sm">Successfully saved!</span>
        </div>
        <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/30">
          <AlertCircle className="h-5 w-5 text-red-500" />
          <span className="text-sm text-red-700 dark:text-red-400">Error occurred</span>
        </div>
      </div>
    ),
  },
  // Input Fields
  {
    id: "input-1",
    content: (
      <div className="w-56 space-y-3 rounded-xl border border-border bg-card p-4 shadow-sm">
        <Input placeholder="Email" className="h-9 text-sm" />
        <Input type="password" placeholder="Password" className="h-9 text-sm" />
      </div>
    ),
  },
  // Checkboxes
  {
    id: "checkbox-1",
    content: (
      <div className="space-y-3 rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <Checkbox id="c1" defaultChecked className="h-5 w-5" />
          <label className="text-sm">Option A</label>
        </div>
        <div className="flex items-center gap-3">
          <Checkbox id="c2" className="h-5 w-5" />
          <label className="text-sm">Option B</label>
        </div>
      </div>
    ),
  },
  // Sidebar
  {
    id: "sidebar-1",
    content: (
      <div className="w-36 rounded-xl border border-border bg-card p-3 shadow-sm">
        <div className="space-y-1">
          <div className="flex items-center gap-2 rounded-lg bg-primary/10 p-2 text-primary">
            <Home className="h-4 w-4" /><span className="text-sm">Home</span>
          </div>
          <div className="flex items-center gap-2 p-2 text-muted-foreground">
            <Users className="h-4 w-4" /><span className="text-sm">Users</span>
          </div>
          <div className="flex items-center gap-2 p-2 text-muted-foreground">
            <Settings className="h-4 w-4" /><span className="text-sm">Settings</span>
          </div>
        </div>
      </div>
    ),
  },
  // Buttons
  {
    id: "buttons-1",
    content: (
      <div className="flex flex-wrap gap-2 rounded-xl border border-border bg-card p-4 shadow-sm">
        <Button size="sm" className="h-9">Primary</Button>
        <Button size="sm" variant="secondary" className="h-9">Secondary</Button>
        <Button size="sm" variant="outline" className="h-9">Outline</Button>
      </div>
    ),
  },
  // Icon Buttons
  {
    id: "icon-buttons-1",
    content: (
      <div className="flex gap-2 rounded-xl border border-border bg-card p-4 shadow-sm">
        <Button size="icon" variant="outline" className="h-10 w-10"><Plus className="h-5 w-5" /></Button>
        <Button size="icon" variant="outline" className="h-10 w-10"><Search className="h-5 w-5" /></Button>
        <Button size="icon" className="h-10 w-10"><Settings className="h-5 w-5" /></Button>
      </div>
    ),
  },
  // Badges
  {
    id: "badges-1",
    content: (
      <div className="flex flex-wrap gap-2 rounded-xl border border-border bg-card p-4 shadow-sm">
        <Badge>Default</Badge>
        <Badge variant="secondary">Secondary</Badge>
        <Badge variant="outline">Outline</Badge>
        <Badge variant="destructive">Error</Badge>
      </div>
    ),
  },
  // Loading
  {
    id: "loading-1",
    content: (
      <div className="flex items-center gap-4 rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="h-8 w-8 animate-spin rounded-full border-3 border-primary border-t-transparent" />
        <div className="space-y-2">
          <div className="h-3 w-24 animate-pulse rounded bg-muted" />
          <div className="h-3 w-20 animate-pulse rounded bg-muted" />
        </div>
      </div>
    ),
  },
  // Empty State
  {
    id: "empty-1",
    content: (
      <div className="rounded-xl border border-border bg-card p-6 text-center shadow-sm">
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-muted">
          <FileText className="h-7 w-7 text-muted-foreground" />
        </div>
        <div className="text-sm font-medium">No items</div>
        <div className="text-xs text-muted-foreground">Get started</div>
      </div>
    ),
  },
  // Avatar Group
  {
    id: "avatars-1",
    content: (
      <div className="flex items-center -space-x-3 rounded-xl border border-border bg-card p-4 shadow-sm">
        <Avatar className="h-12 w-12 border-2 border-card">
          <AvatarFallback>A</AvatarFallback>
        </Avatar>
        <Avatar className="h-12 w-12 border-2 border-card">
          <AvatarFallback>B</AvatarFallback>
        </Avatar>
        <Avatar className="h-12 w-12 border-2 border-card">
          <AvatarFallback>C</AvatarFallback>
        </Avatar>
        <Avatar className="h-12 w-12 border-2 border-card">
          <AvatarFallback className="bg-primary text-white">+3</AvatarFallback>
        </Avatar>
      </div>
    ),
  },
];

// Seeded shuffle for consistent ordering
function shuffleArray<T>(array: T[], seed: number): T[] {
  const shuffled = [...array];
  let currentSeed = seed;

  const random = () => {
    currentSeed = (currentSeed * 9301 + 49297) % 233280;
    return currentSeed / 233280;
  };

  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

interface CarouselRowProps {
  items: typeof componentPreviews;
  direction: "left" | "right";
  speed?: number;
}

function CarouselRow({ items, direction, speed = 50 }: CarouselRowProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState(0);
  const animationRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  // Calculate single set width
  const itemWidth = 280;
  const gap = 24;
  const singleSetWidth = items.length * (itemWidth + gap);

  useEffect(() => {
    const animate = (timestamp: number) => {
      if (!lastTimeRef.current) {
        lastTimeRef.current = timestamp;
      }

      const deltaTime = (timestamp - lastTimeRef.current) / 1000;
      lastTimeRef.current = timestamp;

      setOffset((prev) => {
        let newOffset = prev + speed * deltaTime;
        // Reset seamlessly when we've scrolled one full set
        if (newOffset >= singleSetWidth) {
          newOffset = newOffset % singleSetWidth;
        }
        return newOffset;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [speed, singleSetWidth]);

  // Triple items for seamless loop
  const tripleItems = [...items, ...items, ...items];

  const translateX = direction === "left" ? -offset : offset - singleSetWidth;

  return (
    <div ref={containerRef} className="relative flex overflow-hidden py-3">
      <div
        className="flex gap-6"
        style={{
          transform: `translateX(${translateX}px)`,
          willChange: "transform",
        }}
      >
        {tripleItems.map((item, index) => (
          <div key={`${item.id}-${index}`} className="flex-shrink-0">
            {item.content}
          </div>
        ))}
      </div>
    </div>
  );
}

export function HeroCarousel() {
  const row1Items = shuffleArray(componentPreviews, 12345);
  const row2Items = shuffleArray(componentPreviews, 67890);
  const row3Items = shuffleArray(componentPreviews, 11111);

  return (
    <div className="absolute inset-0 flex flex-col justify-center gap-6 overflow-hidden">
      <CarouselRow items={row1Items} direction="left" speed={40} />
      <CarouselRow items={row2Items} direction="right" speed={35} />
      <CarouselRow items={row3Items} direction="left" speed={45} />
    </div>
  );
}
