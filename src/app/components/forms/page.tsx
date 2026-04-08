"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";
import {
  TextCursorInput,
  ChevronDown,
  CheckSquare,
  LayoutGrid,
} from "lucide-react";

const formsSidebarSections = [
  {
    items: [
      { label: "Input Fields", href: "#inputs", icon: TextCursorInput },
      { label: "Select & Dropdowns", href: "#selects", icon: ChevronDown },
      { label: "Checkboxes & Radios", href: "#checkboxes", icon: CheckSquare },
      { label: "Form Layouts", href: "#layouts", icon: LayoutGrid },
    ],
  },
];

export default function FormsPage() {
  return (
    <TwoColumnLayout
      title="Forms"
      sidebarSections={formsSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          Form components for collecting user input. Includes text fields, selects,
          checkboxes, and complete form layouts.
        </p>
      </div>

      {/* Input Fields */}
      <section id="inputs" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Input Fields</h2>
        <p className="mb-6 text-muted-foreground">
          Text input variations for different use cases.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Input" count={1}>
            <div className="w-full max-w-[200px] space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" placeholder="you@example.com" />
            </div>
          </ComponentCard>

          <ComponentCard title="With Error" count={1}>
            <div className="w-full max-w-[200px] space-y-2">
              <Label htmlFor="error-input">Username</Label>
              <Input id="error-input" className="border-danger-500" defaultValue="invalid" />
              <p className="text-xs text-danger-500">Username already taken</p>
            </div>
          </ComponentCard>

          <ComponentCard title="Disabled" count={1}>
            <div className="w-full max-w-[200px] space-y-2">
              <Label htmlFor="disabled">Disabled</Label>
              <Input id="disabled" disabled placeholder="Cannot edit" />
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Selects */}
      <section id="selects" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Select & Dropdowns</h2>
        <p className="mb-6 text-muted-foreground">
          Dropdown menus for selecting from predefined options.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Select" count={1}>
            <div className="w-full max-w-[200px] space-y-2">
              <Label>Country</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select country" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="us">United States</SelectItem>
                  <SelectItem value="uk">United Kingdom</SelectItem>
                  <SelectItem value="ca">Canada</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Checkboxes */}
      <section id="checkboxes" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Checkboxes & Radios</h2>
        <p className="mb-6 text-muted-foreground">
          Selection controls for single or multiple choices.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Checkboxes" count={1}>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox id="terms" defaultChecked />
                <Label htmlFor="terms">Accept terms</Label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox id="newsletter" />
                <Label htmlFor="newsletter">Subscribe to newsletter</Label>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Radio Buttons" count={1}>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="h-4 w-4 rounded-full border-4 border-primary bg-white" />
                <Label>Option A (Selected)</Label>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-4 w-4 rounded-full border-2 border-border" />
                <Label className="text-muted-foreground">Option B</Label>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Form Layouts */}
      <section id="layouts" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">Form Layouts</h2>
        <p className="mb-6 text-muted-foreground">
          Complete form layouts for common use cases.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Contact Form" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-4">
              <h3 className="mb-3 text-sm font-medium">Contact Us</h3>
              <div className="space-y-3">
                <Input placeholder="Name" className="h-8 text-sm" />
                <Input placeholder="Email" className="h-8 text-sm" />
                <Button size="sm" className="w-full">Submit</Button>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Login Form" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-4">
              <h3 className="mb-3 text-sm font-medium">Sign In</h3>
              <div className="space-y-3">
                <Input placeholder="Email" className="h-8 text-sm" />
                <Input type="password" placeholder="Password" className="h-8 text-sm" />
                <div className="flex items-center gap-2">
                  <Checkbox id="remember" />
                  <Label htmlFor="remember" className="text-xs">Remember me</Label>
                </div>
                <Button size="sm" className="w-full">Sign In</Button>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
