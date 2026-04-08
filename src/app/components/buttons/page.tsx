"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ThreeColumnLayout } from "@/components/three-column-layout";
import { ComponentPreview } from "@/components/component-preview";
import { motion } from "framer-motion";
import { fadeIn, staggerContainer } from "@/lib/animations";
import {
  Plus,
  Download,
  Send,
  Trash2,
  Settings,
  ChevronRight,
  Loader2,
  Check,
  X,
  ArrowRight,
  ExternalLink,
  Circle,
  CircleDot,
  Square,
  EyeOff,
  Link,
  Play,
  MousePointer,
  MousePointerClick,
  Ban,
  Loader,
  Minimize2,
  Maximize2,
  Image,
} from "lucide-react";

const sidebarSections = [
  {
    title: "Variants",
    items: [
      { label: "Primary", href: "#primary", icon: Circle },
      { label: "Secondary", href: "#secondary", icon: CircleDot },
      { label: "Outline", href: "#outline", icon: Square },
      { label: "Ghost", href: "#ghost", icon: EyeOff },
      { label: "Destructive", href: "#destructive", icon: Trash2 },
      { label: "Link", href: "#link", icon: Link },
    ],
  },
  {
    title: "States",
    items: [
      { label: "Default", href: "#states-default", icon: Play },
      { label: "Hover", href: "#states-hover", icon: MousePointer },
      { label: "Active", href: "#states-active", icon: MousePointerClick },
      { label: "Disabled", href: "#states-disabled", icon: Ban },
      { label: "Loading", href: "#states-loading", icon: Loader },
    ],
  },
  {
    title: "Sizes",
    items: [
      { label: "Small", href: "#size-sm", icon: Minimize2 },
      { label: "Default", href: "#size-default", icon: Square },
      { label: "Large", href: "#size-lg", icon: Maximize2 },
      { label: "Icon", href: "#size-icon", icon: Image },
    ],
  },
];

// Documentation content for the right column
function DocumentationContent() {
  return (
    <div className="space-y-6 text-sm">
      <div>
        <h4 className="mb-2 font-semibold">Overview</h4>
        <p className="text-muted-foreground">
          Buttons are interactive elements that trigger actions. They communicate
          what will happen when users interact with them.
        </p>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">When to Use</h4>
        <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
          <li>To trigger an action or event</li>
          <li>To submit forms</li>
          <li>To navigate to another page</li>
          <li>To toggle states or options</li>
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Best Practices</h4>
        <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
          <li>Use clear, action-oriented labels</li>
          <li>Maintain consistent sizing within context</li>
          <li>Use primary buttons sparingly (one per section)</li>
          <li>Provide visual feedback for interactions</li>
          <li>Ensure adequate touch targets (min 44px)</li>
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Accessibility</h4>
        <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
          <li>Use semantic button elements</li>
          <li>Include aria-label for icon-only buttons</li>
          <li>Maintain 4.5:1 contrast ratio</li>
          <li>Support keyboard navigation</li>
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Code Example</h4>
        <pre className="overflow-x-auto rounded-lg bg-muted p-3 text-xs">
{`import { Button } from "@/components/ui/button"

<Button variant="default">
  Click me
</Button>

<Button variant="outline" size="sm">
  Small Outline
</Button>

<Button disabled>
  Disabled
</Button>`}
        </pre>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Props</h4>
        <div className="space-y-2">
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">variant</code>
            <p className="mt-1 text-xs text-muted-foreground">
              "default" | "secondary" | "outline" | "ghost" | "destructive" | "link"
            </p>
          </div>
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">size</code>
            <p className="mt-1 text-xs text-muted-foreground">
              "default" | "sm" | "lg" | "icon"
            </p>
          </div>
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">disabled</code>
            <p className="mt-1 text-xs text-muted-foreground">boolean</p>
          </div>
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">asChild</code>
            <p className="mt-1 text-xs text-muted-foreground">boolean - Render as child element</p>
          </div>
        </div>
      </div>
    </div>
  );
}


export default function ButtonsDetailPage() {
  const [isLoading, setIsLoading] = useState(false);

  return (
    <ThreeColumnLayout
      title="Buttons"
      description="Interactive elements that trigger actions when clicked or tapped."
      sidebarSections={sidebarSections}
      documentationContent={<DocumentationContent />}
    >
      {/* Variants Section */}
      <section id="primary" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Variants</h2>
        <p className="mb-6 text-muted-foreground">
          Different button variants for various use cases and visual hierarchy.
        </p>

        <div className="space-y-6">
          <ComponentPreview
            title="Button Variants"
            code={`<Button>Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="outline">Outline</Button>
<Button variant="ghost">Ghost</Button>
<Button variant="destructive">Destructive</Button>
<Button variant="link">Link</Button>`}
          >
            <div className="flex flex-wrap items-center gap-4">
              <Button>Primary</Button>
              <Button variant="secondary">Secondary</Button>
              <Button variant="outline">Outline</Button>
              <Button variant="ghost">Ghost</Button>
              <Button variant="destructive">Destructive</Button>
              <Button variant="link">Link</Button>
            </div>
          </ComponentPreview>
        </div>
      </section>

      {/* States Section */}
      <section id="states-default" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">States</h2>
        <p className="mb-6 text-muted-foreground">
          Button states provide visual feedback for user interactions.
        </p>

        <div className="space-y-6">
          <ComponentPreview
            title="Interactive States"
            code={`// Default State
<Button>Default</Button>

// Disabled State
<Button disabled>Disabled</Button>

// Loading State
<Button disabled>
  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
  Loading...
</Button>

// Success State
<Button className="bg-green-600 hover:bg-green-700">
  <Check className="mr-2 h-4 w-4" />
  Success
</Button>`}
          >
            <div className="space-y-6">
              <div>
                <p className="mb-3 text-sm font-medium">Default State</p>
                <div className="flex flex-wrap items-center gap-4">
                  <Button>Default</Button>
                  <Button variant="secondary">Secondary</Button>
                  <Button variant="outline">Outline</Button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm font-medium">Disabled State</p>
                <div className="flex flex-wrap items-center gap-4">
                  <Button disabled>Disabled</Button>
                  <Button variant="secondary" disabled>Disabled</Button>
                  <Button variant="outline" disabled>Disabled</Button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm font-medium">Loading State</p>
                <div className="flex flex-wrap items-center gap-4">
                  <Button disabled>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Loading...
                  </Button>
                  <Button variant="secondary" disabled>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing
                  </Button>
                  <Button variant="outline" disabled>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Please wait
                  </Button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm font-medium">Success State</p>
                <div className="flex flex-wrap items-center gap-4">
                  <Button className="bg-green-600 hover:bg-green-700">
                    <Check className="mr-2 h-4 w-4" />
                    Success
                  </Button>
                </div>
              </div>
            </div>
          </ComponentPreview>
        </div>
      </section>

      {/* Sizes Section */}
      <section id="size-sm" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Sizes</h2>
        <p className="mb-6 text-muted-foreground">
          Multiple sizes to fit different contexts and layouts.
        </p>

        <div className="space-y-6">
          <ComponentPreview
            title="Size Comparison"
            code={`<Button size="sm">Small</Button>
<Button>Default</Button>
<Button size="lg">Large</Button>
<Button size="icon">
  <Plus className="h-4 w-4" />
</Button>`}
          >
            <div className="flex flex-wrap items-end gap-4">
              <div className="text-center">
                <Button size="sm">Small</Button>
                <p className="mt-2 text-xs text-muted-foreground">size=&quot;sm&quot;</p>
              </div>
              <div className="text-center">
                <Button>Default</Button>
                <p className="mt-2 text-xs text-muted-foreground">default</p>
              </div>
              <div className="text-center">
                <Button size="lg">Large</Button>
                <p className="mt-2 text-xs text-muted-foreground">size=&quot;lg&quot;</p>
              </div>
              <div className="text-center">
                <Button size="icon">
                  <Plus className="h-4 w-4" />
                </Button>
                <p className="mt-2 text-xs text-muted-foreground">size=&quot;icon&quot;</p>
              </div>
            </div>
          </ComponentPreview>

          <ComponentPreview
            title="Icon Buttons"
            code={`<Button size="icon" variant="outline">
  <Plus className="h-4 w-4" />
</Button>
<Button size="icon" variant="destructive">
  <Trash2 className="h-4 w-4" />
</Button>
<Button size="icon" variant="ghost">
  <X className="h-4 w-4" />
</Button>`}
          >
            <div className="flex flex-wrap items-center gap-3">
              <Button size="icon" variant="outline">
                <Plus className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="outline">
                <Settings className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="outline">
                <Download className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="outline">
                <Send className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="destructive">
                <Trash2 className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost">
                <X className="h-4 w-4" />
              </Button>
            </div>
          </ComponentPreview>
        </div>
      </section>

      {/* With Icons Section */}
      <section id="with-icons" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">With Icons</h2>
        <p className="mb-6 text-muted-foreground">
          Buttons can include icons to enhance visual communication.
        </p>

        <div className="space-y-6">
          <ComponentPreview
            title="Icon Positions"
            code={`// Leading Icon
<Button>
  <Plus className="mr-2 h-4 w-4" />
  Add Item
</Button>

// Trailing Icon
<Button>
  Continue
  <ArrowRight className="ml-2 h-4 w-4" />
</Button>`}
          >
            <div className="space-y-4">
              <div>
                <p className="mb-3 text-sm font-medium">Leading Icon</p>
                <div className="flex flex-wrap items-center gap-4">
                  <Button>
                    <Plus className="mr-2 h-4 w-4" />
                    Add Item
                  </Button>
                  <Button variant="secondary">
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </Button>
                  <Button variant="outline">
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </Button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm font-medium">Trailing Icon</p>
                <div className="flex flex-wrap items-center gap-4">
                  <Button>
                    Continue
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                  <Button variant="secondary">
                    Learn More
                    <ExternalLink className="ml-2 h-4 w-4" />
                  </Button>
                  <Button variant="outline">
                    Next Step
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </ComponentPreview>
        </div>
      </section>

      {/* Button Groups Section */}
      <section id="button-groups" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Button Groups</h2>
        <p className="mb-6 text-muted-foreground">
          Combine buttons for related actions or toggle options.
        </p>

        <div className="space-y-6">
          <ComponentPreview
            title="Action Groups"
            code={`// Primary + Secondary
<Button>Save Changes</Button>
<Button variant="outline">Cancel</Button>

// Destructive Action
<Button variant="destructive">Delete</Button>
<Button variant="outline">Cancel</Button>

// Segmented Control
<div className="inline-flex rounded-lg border p-1">
  <Button variant="secondary" size="sm">Day</Button>
  <Button variant="ghost" size="sm">Week</Button>
  <Button variant="ghost" size="sm">Month</Button>
</div>`}
          >
            <div className="space-y-6">
              <div>
                <p className="mb-3 text-sm font-medium">Primary + Secondary</p>
                <div className="flex items-center gap-3">
                  <Button>Save Changes</Button>
                  <Button variant="outline">Cancel</Button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm font-medium">Destructive Action</p>
                <div className="flex items-center gap-3">
                  <Button variant="destructive">Delete</Button>
                  <Button variant="outline">Cancel</Button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm font-medium">Segmented Control</p>
                <div className="inline-flex rounded-lg border p-1">
                  <Button variant="secondary" size="sm" className="rounded-md">
                    Day
                  </Button>
                  <Button variant="ghost" size="sm" className="rounded-md">
                    Week
                  </Button>
                  <Button variant="ghost" size="sm" className="rounded-md">
                    Month
                  </Button>
                </div>
              </div>
            </div>
          </ComponentPreview>
        </div>
      </section>

      {/* Real-world Examples */}
      <section id="examples" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Real-world Examples</h2>
        <p className="mb-6 text-muted-foreground">
          How buttons are used in common UI patterns.
        </p>

        <div className="space-y-6">
          <ComponentPreview
            title="Form Actions"
            code={`<Card className="max-w-md">
  <CardHeader>
    <CardTitle>Create Account</CardTitle>
  </CardHeader>
  <CardContent className="space-y-4">
    <Input placeholder="Email" />
    <Input type="password" placeholder="Password" />
    <div className="flex gap-3">
      <Button className="flex-1">Sign Up</Button>
      <Button variant="outline">Cancel</Button>
    </div>
  </CardContent>
</Card>`}
          >
            <Card className="max-w-md">
              <CardHeader>
                <CardTitle className="text-lg">Create Account</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Email</label>
                  <input
                    type="email"
                    placeholder="you@example.com"
                    className="w-full rounded-md border px-3 py-2 text-sm"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Password</label>
                  <input
                    type="password"
                    placeholder="Enter password"
                    className="w-full rounded-md border px-3 py-2 text-sm"
                  />
                </div>
                <div className="flex gap-3 pt-2">
                  <Button className="flex-1">Sign Up</Button>
                  <Button variant="outline">Cancel</Button>
                </div>
              </CardContent>
            </Card>
          </ComponentPreview>

          <ComponentPreview
            title="Card Actions"
            code={`<Card className="max-w-sm">
  <CardHeader>
    <Badge className="mb-2 w-fit">Pro Plan</Badge>
    <CardTitle>Upgrade to Pro</CardTitle>
  </CardHeader>
  <CardContent>
    <p className="mb-4 text-sm text-muted-foreground">
      Get unlimited access to all features.
    </p>
    <Button className="w-full">
      Upgrade Now
      <ArrowRight className="ml-2 h-4 w-4" />
    </Button>
  </CardContent>
</Card>`}
          >
            <Card className="max-w-sm">
              <CardHeader>
                <Badge className="mb-2 w-fit">Pro Plan</Badge>
                <CardTitle>Upgrade to Pro</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-4 text-sm text-muted-foreground">
                  Get unlimited access to all features and priority support.
                </p>
                <Button className="w-full">
                  Upgrade Now
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </ComponentPreview>
        </div>
      </section>
    </ThreeColumnLayout>
  );
}
