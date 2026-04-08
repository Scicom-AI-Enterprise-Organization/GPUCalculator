"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertCircle,
  Info,
  Check,
  ChevronRight,
  Settings,
  Search,
  Plus,
  TrendingUp,
  Users,
  DollarSign,
  X,
  FileText,
  Menu,
  Home,
  Layers,
  MessageCircle,
  Bot,
  Send,
  Mic,
  Volume2,
  Sparkles,
  Headphones,
  ListChecks,
} from "lucide-react";
import { motion } from "framer-motion";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";
import { SectionHeader } from "@/components/section-header";
import { staggerContainerFast } from "@/lib/animations";
import {
  Filter,
  SlidersHorizontal,
  BarChart3,
  LayoutGrid,
  FormInput,
  Compass,
  MousePointerClick,
} from "lucide-react";

const componentsSidebarSections = [
  {
    items: [
      { label: "AIMS Filter", href: "#aims-filter", icon: Filter },
      { label: "Data Visualization", href: "#data-visualization", icon: BarChart3 },
      { label: "Cards", href: "#cards", icon: LayoutGrid },
      { label: "Overlays", href: "#overlays", icon: Layers },
      { label: "Forms", href: "#forms", icon: FormInput },
      { label: "Navigation", href: "#navigation", icon: Compass },
      { label: "Buttons & Actions", href: "#buttons", icon: MousePointerClick },
      { label: "Feedback", href: "#feedback", icon: MessageCircle },
      { label: "AI", href: "#ai", icon: Bot },
    ],
  },
];

export default function ComponentsPage() {
  return (
    <TwoColumnLayout
      title="Components"
      sidebarSections={componentsSidebarSections}
    >
      {/* AIMS Filter */}
      <section id="aims-filter" className="mb-12">
        <SectionHeader
          title="AIMS Filter"
          description="A comprehensive filter panel with Standard and Advanced modes for filtering sessions, users, time ranges, sentiment, and more."
          href="/components/aims-filter"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Standard Mode */}
          <ComponentCard title="Standard Mode" count={6} href="/components/aims-filter">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 flex items-center gap-2">
                <Filter className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">Filter Panel</span>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="h-6 w-20 rounded bg-muted" />
                  <div className="h-6 w-16 rounded bg-muted" />
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-6 w-24 rounded bg-muted" />
                  <div className="h-6 w-12 rounded bg-primary/20" />
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* Advanced Mode */}
          <ComponentCard title="Advanced Mode" count={8} href="/components/aims-filter">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">Advanced Filters</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="h-5 rounded bg-muted" />
                <div className="h-5 rounded bg-muted" />
                <div className="h-5 rounded bg-primary/20" />
                <div className="h-5 rounded bg-muted" />
              </div>
              <div className="mt-2 flex justify-end gap-1">
                <div className="h-5 w-12 rounded bg-muted" />
                <div className="h-5 w-12 rounded bg-primary" />
              </div>
            </div>
          </ComponentCard>

          {/* Filter Chips */}
          <ComponentCard title="Filter Chips" count={4} href="/components/aims-filter">
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary" className="gap-1">
                Sentiment: Positive
                <X className="h-3 w-3" />
              </Badge>
              <Badge variant="secondary" className="gap-1">
                Date: Last 7 days
                <X className="h-3 w-3" />
              </Badge>
              <Badge variant="outline" className="text-xs">
                + Add Filter
              </Badge>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Data Visualization */}
      <section id="data-visualization" className="mb-12">
        <SectionHeader
          title="Data Visualization"
          href="/components/data-visualization"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Charts & Graphs */}
          <ComponentCard title="Charts & Graphs" count={12} href="/components/data-visualization/charts">
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Revenue</span>
                <Badge variant="secondary" className="h-5 text-xs">
                  +12%
                </Badge>
              </div>
              <div className="flex h-16 items-end gap-1">
                {[40, 65, 45, 80, 55, 90, 70].map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t bg-primary/80"
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
            </div>
          </ComponentCard>

          {/* Stats Cards */}
          <ComponentCard title="Stats Cards" count={8} href="/components/data-visualization/stats">
            <div className="flex gap-2">
              <div className="rounded-lg border border-border bg-card p-3">
                <DollarSign className="mb-1 h-4 w-4 text-primary" />
                <div className="text-lg font-bold">$45.2k</div>
                <div className="text-xs text-muted-foreground">Revenue</div>
              </div>
              <div className="rounded-lg border border-border bg-card p-3">
                <Users className="mb-1 h-4 w-4 text-success-500" />
                <div className="text-lg font-bold">2,340</div>
                <div className="text-xs text-muted-foreground">Users</div>
              </div>
            </div>
          </ComponentCard>

          {/* Progress Indicators */}
          <ComponentCard title="Progress Indicators" count={6} href="/components/data-visualization/progress">
            <div className="w-full max-w-[160px] space-y-3">
              <div>
                <div className="mb-1 flex justify-between text-xs">
                  <span>Sales</span>
                  <span>78%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div className="h-2 w-[78%] rounded-full bg-primary" />
                </div>
              </div>
              <div>
                <div className="mb-1 flex justify-between text-xs">
                  <span>Growth</span>
                  <span>45%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div className="h-2 w-[45%] rounded-full bg-success-500" />
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* KPI Widgets */}
          <ComponentCard title="KPI Widgets" count={5} href="/components/data-visualization/kpi">
            <div className="rounded-lg border border-border bg-card p-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-muted-foreground">
                    Conversion
                  </div>
                  <div className="text-xl font-bold">3.24%</div>
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-success-100 dark:bg-success-900/30">
                  <TrendingUp className="h-5 w-5 text-success-600" />
                </div>
              </div>
              <div className="mt-2 text-xs text-success-600">
                +0.5% from last week
              </div>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Cards */}
      <section id="cards" className="mb-12">
        <SectionHeader
          title="Cards"
          href="/components/cards"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Basic Cards */}
          <ComponentCard title="Basic Cards" count={4} href="/components/cards/basic">
            <Card className="w-full max-w-[180px]">
              <CardHeader className="p-3">
                <CardTitle className="text-sm">Card Title</CardTitle>
                <CardDescription className="text-xs">
                  Description text
                </CardDescription>
              </CardHeader>
              <CardContent className="p-3 pt-0">
                <p className="text-xs text-muted-foreground">
                  Card content goes here.
                </p>
              </CardContent>
            </Card>
          </ComponentCard>

          {/* Profile Cards */}
          <ComponentCard title="Profile Cards" count={6} href="/components/cards/profile">
            <div className="rounded-lg border border-border bg-card p-4 text-center">
              <Avatar className="mx-auto mb-2 h-12 w-12">
                <AvatarFallback>JD</AvatarFallback>
              </Avatar>
              <div className="text-sm font-medium">John Doe</div>
              <div className="text-xs text-muted-foreground">
                Product Designer
              </div>
            </div>
          </ComponentCard>

          {/* Pricing Cards */}
          <ComponentCard title="Pricing Cards" count={3} href="/components/cards/pricing">
            <div className="rounded-lg border-2 border-primary bg-card p-3">
              <Badge className="mb-2">Popular</Badge>
              <div className="text-xl font-bold">
                $29
                <span className="text-sm font-normal text-muted-foreground">
                  /mo
                </span>
              </div>
              <div className="mt-2 space-y-1">
                <div className="flex items-center gap-1 text-xs">
                  <Check className="h-3 w-3 text-success-500" />
                  <span>10 Projects</span>
                </div>
                <div className="flex items-center gap-1 text-xs">
                  <Check className="h-3 w-3 text-success-500" />
                  <span>Unlimited Users</span>
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* Feature Cards */}
          <ComponentCard title="Feature Cards" count={4} href="/components/cards/feature">
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                <Layers className="h-4 w-4 text-primary" />
              </div>
              <div className="text-sm font-medium">Feature Title</div>
              <div className="text-xs text-muted-foreground">
                Brief description of the feature.
              </div>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Overlays */}
      <section id="overlays" className="mb-12">
        <SectionHeader
          title="Overlays"
          href="/components/overlays"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Modals */}
          <ComponentCard title="Modals & Dialogs" count={5} href="/components/overlays/modals">
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card shadow-lg">
              <div className="flex items-center justify-between border-b border-border p-2">
                <span className="text-xs font-medium">Dialog Title</span>
                <X className="h-3 w-3 text-muted-foreground" />
              </div>
              <div className="p-3">
                <p className="text-xs text-muted-foreground">
                  Dialog content here...
                </p>
              </div>
              <div className="flex justify-end gap-2 border-t border-border p-2">
                <Button size="sm" variant="outline" className="h-6 text-xs">
                  Cancel
                </Button>
                <Button size="sm" className="h-6 text-xs">
                  Confirm
                </Button>
              </div>
            </div>
          </ComponentCard>

          {/* Tooltips */}
          <ComponentCard title="Tooltips" count={3} href="/components/overlays/tooltips">
            <div className="flex flex-col items-center gap-2">
              <div className="rounded bg-foreground px-2 py-1 text-xs text-background">
                Tooltip text
                <div className="absolute -bottom-1 left-1/2 h-2 w-2 -translate-x-1/2 rotate-45 bg-foreground" />
              </div>
              <Button size="sm" variant="outline">
                Hover me
              </Button>
            </div>
          </ComponentCard>

          {/* Popovers */}
          <ComponentCard title="Popovers" count={4} href="/components/overlays/popovers">
            <div className="relative">
              <div className="absolute -top-16 left-1/2 w-36 -translate-x-1/2 rounded-lg border border-border bg-card p-2 shadow-lg">
                <div className="text-xs font-medium">Popover Title</div>
                <div className="text-xs text-muted-foreground">
                  Additional info here.
                </div>
              </div>
              <Button size="sm" variant="outline">
                Click me
              </Button>
            </div>
          </ComponentCard>

          {/* Toasts */}
          <ComponentCard title="Toasts & Notifications" count={6} href="/components/overlays/toasts">
            <div className="w-full max-w-[200px] space-y-2">
              <div className="flex items-center gap-2 rounded-lg border border-border bg-card p-2 shadow-sm">
                <Check className="h-4 w-4 text-success-500" />
                <span className="text-xs">Successfully saved!</span>
              </div>
              <div className="flex items-center gap-2 rounded-lg border border-danger-200 bg-danger-50 p-2 dark:border-danger-800 dark:bg-danger-900/30">
                <AlertCircle className="h-4 w-4 text-danger-500" />
                <span className="text-xs text-danger-700 dark:text-danger-400">
                  Error occurred
                </span>
              </div>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Forms */}
      <section id="forms" className="mb-12">
        <SectionHeader
          title="Forms"
          href="/components/forms"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Input Fields */}
          <ComponentCard title="Input Fields" count={8} href="/components/forms/inputs">
            <div className="w-full max-w-[180px] space-y-2">
              <div className="space-y-1">
                <Label className="text-xs">Email</Label>
                <Input placeholder="you@example.com" className="h-8 text-sm" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Password</Label>
                <Input
                  type="password"
                  placeholder="••••••••"
                  className="h-8 text-sm"
                />
              </div>
            </div>
          </ComponentCard>

          {/* Select & Dropdowns */}
          <ComponentCard title="Select & Dropdowns" count={5} href="/components/forms/selects">
            <div className="w-full max-w-[180px] space-y-2">
              <Select>
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue placeholder="Select option" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Option 1</SelectItem>
                  <SelectItem value="2">Option 2</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </ComponentCard>

          {/* Checkboxes & Radios */}
          <ComponentCard title="Checkboxes & Radios" count={4} href="/components/forms/checkboxes">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Checkbox id="c1" defaultChecked />
                <Label htmlFor="c1" className="text-sm">
                  Option A
                </Label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox id="c2" />
                <Label htmlFor="c2" className="text-sm">
                  Option B
                </Label>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-4 w-4 rounded-full border-4 border-primary bg-white" />
                <Label className="text-sm">Selected</Label>
              </div>
            </div>
          </ComponentCard>

          {/* Form Layouts */}
          <ComponentCard title="Form Layouts" count={3} href="/components/forms/layouts">
            <div className="w-full max-w-[180px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 text-xs font-medium">Contact Form</div>
              <div className="space-y-2">
                <Input placeholder="Name" className="h-6 text-xs" />
                <Input placeholder="Email" className="h-6 text-xs" />
                <Button size="sm" className="h-6 w-full text-xs">
                  Submit
                </Button>
              </div>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Navigation */}
      <section id="navigation" className="mb-12">
        <SectionHeader
          title="Navigation"
          href="/components/navigation"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Navbars */}
          <ComponentCard title="Navbars" count={4} href="/components/navigation/navbars">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card">
              <div className="flex items-center justify-between p-2">
                <div className="flex items-center gap-2">
                  <div className="h-5 w-5 rounded bg-primary" />
                  <span className="text-xs font-medium">Logo</span>
                </div>
                <Menu className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>
          </ComponentCard>

          {/* Sidebars */}
          <ComponentCard title="Sidebars" count={3} href="/components/navigation/sidebars">
            <div className="w-24 rounded-lg border border-border bg-card p-2">
              <div className="space-y-1">
                <div className="flex items-center gap-2 rounded bg-primary/10 p-1.5 text-primary">
                  <Home className="h-3 w-3" />
                  <span className="text-xs">Home</span>
                </div>
                <div className="flex items-center gap-2 p-1.5 text-muted-foreground">
                  <Users className="h-3 w-3" />
                  <span className="text-xs">Users</span>
                </div>
                <div className="flex items-center gap-2 p-1.5 text-muted-foreground">
                  <Settings className="h-3 w-3" />
                  <span className="text-xs">Settings</span>
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* Breadcrumbs */}
          <ComponentCard title="Breadcrumbs" count={2} href="/components/navigation/breadcrumbs">
            <div className="flex items-center gap-1 text-sm">
              <span className="text-muted-foreground">Home</span>
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Products</span>
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <span className="font-medium">Details</span>
            </div>
          </ComponentCard>

          {/* Tabs */}
          <ComponentCard title="Tabs" count={3} href="/components/navigation/tabs">
            <div className="flex rounded-lg bg-muted p-1">
              <button className="rounded-md bg-card px-3 py-1 text-xs font-medium shadow-sm">
                Tab 1
              </button>
              <button className="px-3 py-1 text-xs text-muted-foreground">
                Tab 2
              </button>
              <button className="px-3 py-1 text-xs text-muted-foreground">
                Tab 3
              </button>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Buttons & Actions */}
      <section id="buttons" className="mb-12">
        <SectionHeader
          title="Buttons & Actions"
          href="/components/buttons"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Button Variants */}
          <ComponentCard title="Button Variants" count={6} href="/components/buttons">
            <div className="flex flex-wrap justify-center gap-2">
              <Button size="sm">Primary</Button>
              <Button size="sm" variant="secondary">
                Secondary
              </Button>
              <Button size="sm" variant="outline">
                Outline
              </Button>
            </div>
          </ComponentCard>

          {/* Icon Buttons */}
          <ComponentCard title="Icon Buttons" count={4} href="/components/buttons/icons">
            <div className="flex gap-2">
              <Button size="icon" variant="outline" className="h-8 w-8">
                <Plus className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="outline" className="h-8 w-8">
                <Search className="h-4 w-4" />
              </Button>
              <Button size="icon" className="h-8 w-8">
                <Settings className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="destructive" className="h-8 w-8">
                <X className="h-4 w-4" />
              </Button>
            </div>
          </ComponentCard>

          {/* Button Groups */}
          <ComponentCard title="Button Groups" count={2} href="/components/buttons/groups">
            <div className="flex">
              <Button
                size="sm"
                variant="outline"
                className="rounded-r-none border-r-0"
              >
                Left
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="rounded-none border-r-0"
              >
                Middle
              </Button>
              <Button size="sm" variant="outline" className="rounded-l-none">
                Right
              </Button>
            </div>
          </ComponentCard>

          {/* Badges & Tags */}
          <ComponentCard title="Badges & Tags" count={5} href="/components/buttons/badges">
            <div className="flex flex-wrap justify-center gap-2">
              <Badge>Default</Badge>
              <Badge variant="secondary">Secondary</Badge>
              <Badge variant="outline">Outline</Badge>
              <Badge variant="destructive">Error</Badge>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* Feedback */}
      <section id="feedback" className="mb-12">
        <SectionHeader
          title="Feedback"
          href="/components/feedback"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Alerts */}
          <ComponentCard title="Alerts" count={4} href="/components/feedback/alerts">
            <div className="w-full max-w-[200px] space-y-2">
              <Alert className="py-2">
                <Info className="h-3 w-3" />
                <AlertDescription className="text-xs">
                  Info message
                </AlertDescription>
              </Alert>
            </div>
          </ComponentCard>

          {/* Loading States */}
          <ComponentCard title="Loading States" count={4} href="/components/feedback/loading">
            <div className="flex items-center gap-4">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <div className="space-y-1">
                <div className="h-2 w-20 animate-pulse rounded bg-muted" />
                <div className="h-2 w-16 animate-pulse rounded bg-muted" />
              </div>
            </div>
          </ComponentCard>

          {/* Empty States */}
          <ComponentCard title="Empty States" count={3} href="/components/feedback/empty">
            <div className="text-center">
              <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-muted">
                <FileText className="h-5 w-5 text-muted-foreground" />
              </div>
              <div className="text-xs font-medium">No items</div>
              <div className="text-xs text-muted-foreground">
                Get started by adding one
              </div>
            </div>
          </ComponentCard>

          {/* Avatars */}
          <ComponentCard title="Avatars" count={4} href="/components/feedback/avatars">
            <div className="flex items-center -space-x-2">
              <Avatar className="h-8 w-8 border-2 border-background">
                <AvatarFallback className="text-xs">A</AvatarFallback>
              </Avatar>
              <Avatar className="h-8 w-8 border-2 border-background">
                <AvatarFallback className="text-xs">B</AvatarFallback>
              </Avatar>
              <Avatar className="h-8 w-8 border-2 border-background">
                <AvatarFallback className="text-xs">C</AvatarFallback>
              </Avatar>
              <Avatar className="h-8 w-8 border-2 border-background">
                <AvatarFallback className="bg-primary text-xs text-white">
                  +3
                </AvatarFallback>
              </Avatar>
            </div>
          </ComponentCard>
        </motion.div>
      </section>

      {/* AI */}
      <section id="ai" className="mb-0">
        <SectionHeader
          title="AI"
          href="/components/ai"
        />
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainerFast}
          className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3"
        >
          {/* Traditional Chatbot */}
          <ComponentCard title="Traditional Chatbot" count={3} href="/components/ai/chatbot">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-b border-border p-2">
                <Bot className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">AI Assistant</span>
              </div>
              <div className="space-y-2 p-2">
                <div className="flex gap-2">
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <Bot className="h-3 w-3 text-primary" />
                  </div>
                  <div className="rounded-lg bg-muted px-2 py-1 text-xs">
                    How can I help you today?
                  </div>
                </div>
                <div className="flex justify-end">
                  <div className="rounded-lg bg-primary px-2 py-1 text-xs text-white">
                    Show me reports
                  </div>
                </div>
              </div>
              <div className="flex gap-1 border-t border-border p-2">
                <Input placeholder="Type a message..." className="h-6 text-xs" />
                <Button size="icon" className="h-6 w-6">
                  <Send className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </ComponentCard>

          {/* Structured Chatbot (with leading questions) */}
          <ComponentCard title="Structured Chatbot" count={4} href="/components/ai/structured">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-b border-border p-2">
                <ListChecks className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">Guided Assistant</span>
              </div>
              <div className="space-y-2 p-2">
                <div className="rounded-lg bg-muted px-2 py-1 text-xs">
                  What would you like to do?
                </div>
                <div className="space-y-1">
                  <button className="w-full rounded border border-primary/30 bg-primary/5 px-2 py-1 text-left text-xs text-primary hover:bg-primary/10">
                    Create a new report
                  </button>
                  <button className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted">
                    View analytics
                  </button>
                  <button className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted">
                    Get help
                  </button>
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* Headless Chatbot (AG UI style) */}
          <ComponentCard title="Headless Chatbot" count={5} href="/components/ai/headless">
            <div className="w-full max-w-[200px] space-y-2">
              <div className="rounded-lg border border-border bg-card p-2">
                <div className="mb-2 flex items-center gap-1">
                  <Sparkles className="h-3 w-3 text-primary" />
                  <span className="text-xs font-medium">AI Generated</span>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>Revenue</span>
                    <span className="font-medium">$45.2k</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted">
                    <div className="h-1.5 w-[75%] rounded-full bg-primary" />
                  </div>
                </div>
              </div>
              <div className="rounded-lg border border-primary/30 bg-primary/5 p-2">
                <div className="flex items-start gap-2">
                  <Bot className="h-4 w-4 shrink-0 text-primary" />
                  <p className="text-xs text-primary">
                    Based on your request, here&apos;s the revenue breakdown.
                  </p>
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* Agent Assist */}
          <ComponentCard title="Agent Assist" count={4} href="/components/ai/agent-assist">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center justify-between border-b border-border p-2">
                <div className="flex items-center gap-2">
                  <Headphones className="h-4 w-4 text-primary" />
                  <span className="text-xs font-medium">Agent Assist</span>
                </div>
                <Badge variant="secondary" className="h-4 text-[10px]">Live</Badge>
              </div>
              <div className="p-2">
                <div className="mb-2 rounded bg-warning-50 p-1.5 dark:bg-warning-900/30">
                  <div className="flex items-center gap-1 text-[10px] font-medium text-warning-700 dark:text-warning-400">
                    <Sparkles className="h-3 w-3" />
                    Suggested Response
                  </div>
                  <p className="mt-1 text-xs text-warning-600 dark:text-warning-300">
                    I can help reset your password...
                  </p>
                </div>
                <div className="flex gap-1">
                  <Button size="sm" className="h-5 flex-1 text-[10px]">
                    Use
                  </Button>
                  <Button size="sm" variant="outline" className="h-5 flex-1 text-[10px]">
                    Edit
                  </Button>
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* Audio Response */}
          <ComponentCard title="Audio Response" count={3} href="/components/ai/audio">
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-3 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4 text-primary" />
                  <span className="text-xs font-medium">Voice Response</span>
                </div>
              </div>
              <div className="mb-3 flex items-center gap-1">
                {[12, 18, 10, 22, 14, 20, 8, 16, 24, 11].map((h, i) => (
                  <div
                    key={i}
                    className="w-1 rounded-full bg-primary/60"
                    style={{ height: `${h}px` }}
                  />
                ))}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">0:12 / 0:45</span>
                <div className="flex gap-1">
                  <Button size="icon" variant="outline" className="h-6 w-6">
                    <Mic className="h-3 w-3" />
                  </Button>
                  <Button size="icon" className="h-6 w-6">
                    <Volume2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </div>
          </ComponentCard>

          {/* AI Chat Bubble */}
          <ComponentCard title="AI Message Bubbles" count={3} href="/components/ai/message-bubbles">
            <div className="w-full max-w-[200px] space-y-2">
              <div className="flex gap-2">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary">
                  <Bot className="h-3 w-3 text-white" />
                </div>
                <div className="space-y-1">
                  <div className="rounded-lg rounded-tl-none bg-muted px-2 py-1 text-xs">
                    I found 3 matching results.
                  </div>
                  <div className="flex gap-1">
                    <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                      helpful
                    </span>
                    <span className="rounded bg-muted px-1.5 py-0.5 text-[10px]">
                      not helpful
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex justify-end">
                <div className="rounded-lg rounded-tr-none bg-primary px-2 py-1 text-xs text-white">
                  Thanks! Show me more details
                </div>
              </div>
            </div>
          </ComponentCard>
        </motion.div>
      </section>
    </TwoColumnLayout>
  );
}
