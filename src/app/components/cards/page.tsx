"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ThreeColumnLayout } from "@/components/three-column-layout";
import { ComponentPreview } from "@/components/component-preview";
import { motion } from "framer-motion";
import { fadeIn, staggerContainer } from "@/lib/animations";
import {
  Check,
  ArrowRight,
  Star,
  Heart,
  MoreHorizontal,
  MapPin,
  Calendar,
  Users,
  TrendingUp,
  Layers,
  Zap,
  Shield,
  Square,
  BarChart3,
  User,
  CreditCard,
  ShoppingBag,
  ImageIcon,
  MousePointerClick,
  PanelRight,
  BoxSelect,
} from "lucide-react";

const sidebarSections = [
  {
    title: "Card Types",
    items: [
      { label: "Basic Card", href: "#basic", icon: Square },
      { label: "Stats Card", href: "#stats", icon: BarChart3 },
      { label: "Profile Card", href: "#profile", icon: User },
      { label: "Pricing Card", href: "#pricing", icon: CreditCard },
      { label: "Feature Card", href: "#feature", icon: Star },
      { label: "Product Card", href: "#product", icon: ShoppingBag },
    ],
  },
  {
    title: "Variations",
    items: [
      { label: "With Image", href: "#with-image", icon: ImageIcon },
      { label: "With Actions", href: "#with-actions", icon: MousePointerClick },
      { label: "Horizontal", href: "#horizontal", icon: PanelRight },
      { label: "Bordered", href: "#bordered", icon: BoxSelect },
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
          Cards are containers that group related content and actions. They provide
          a flexible and extensible content container with multiple variants.
        </p>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">When to Use</h4>
        <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
          <li>To display content in a digestible format</li>
          <li>To group related information together</li>
          <li>To create visual hierarchy on a page</li>
          <li>For dashboard widgets and data summaries</li>
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Best Practices</h4>
        <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
          <li>Keep content concise and scannable</li>
          <li>Use consistent card sizes in grids</li>
          <li>Limit actions to 1-2 primary options</li>
          <li>Use appropriate elevation for hierarchy</li>
          <li>Ensure touch targets are adequate size</li>
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Anatomy</h4>
        <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
          <li><strong>CardHeader</strong> - Title and description</li>
          <li><strong>CardContent</strong> - Main content area</li>
          <li><strong>CardFooter</strong> - Actions and metadata</li>
        </ul>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Code Example</h4>
        <pre className="overflow-x-auto rounded-lg bg-muted p-3 text-xs">
{`import { Card, CardHeader, CardTitle,
  CardDescription, CardContent,
  CardFooter } from "@/components/ui/card"

<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>
      Card description here
    </CardDescription>
  </CardHeader>
  <CardContent>
    <p>Card content</p>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>`}
        </pre>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Components</h4>
        <div className="space-y-2">
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">Card</code>
            <p className="mt-1 text-xs text-muted-foreground">
              Main container component
            </p>
          </div>
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">CardHeader</code>
            <p className="mt-1 text-xs text-muted-foreground">
              Contains title and description
            </p>
          </div>
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">CardContent</code>
            <p className="mt-1 text-xs text-muted-foreground">
              Main content area with padding
            </p>
          </div>
          <div className="rounded-lg border p-2">
            <code className="text-xs font-medium">CardFooter</code>
            <p className="mt-1 text-xs text-muted-foreground">
              Footer for actions, uses flex layout
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}


export default function CardsDetailPage() {
  return (
    <ThreeColumnLayout
      title="Cards"
      description="Flexible containers for grouping related content and actions."
      sidebarSections={sidebarSections}
      documentationContent={<DocumentationContent />}
    >
      {/* Basic Cards Section */}
      <section id="basic" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Basic Cards</h2>
        <p className="mb-6 text-muted-foreground">
          Simple card layouts with header, content, and footer sections.
        </p>

        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="space-y-6"
        >
          <ComponentPreview title="Light Background">
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Basic Card</CardTitle>
                  <CardDescription>A simple card with title and description.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    This is the main content area of the card. You can put any content here.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Card with Footer</CardTitle>
                  <CardDescription>Includes action buttons in the footer.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Content goes here with actions below.
                  </p>
                </CardContent>
                <CardFooter className="gap-2">
                  <Button size="sm">Save</Button>
                  <Button size="sm" variant="outline">Cancel</Button>
                </CardFooter>
              </Card>
            </div>
          </ComponentPreview>

          <ComponentPreview title="Dark Background">
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border-border bg-muted">
                <CardHeader>
                  <CardTitle className="text-white">Dark Card</CardTitle>
                  <CardDescription className="text-muted-foreground">
                    Card styled for dark backgrounds.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Content adapts to dark mode styling.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border bg-muted">
                <CardHeader>
                  <CardTitle className="text-white">With Actions</CardTitle>
                  <CardDescription className="text-muted-foreground">
                    Dark card with footer actions.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Buttons adapt to dark theme.
                  </p>
                </CardContent>
                <CardFooter className="gap-2">
                  <Button size="sm">Primary</Button>
                  <Button size="sm" variant="outline" className="border-border text-muted-foreground">
                    Secondary
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </ComponentPreview>
        </motion.div>
      </section>

      {/* Stats Cards Section */}
      <section id="stats" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Stats Cards</h2>
        <p className="mb-6 text-muted-foreground">
          Display key metrics and statistics in a compact format.
        </p>

        <ComponentPreview title="Dashboard Stats">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Total Revenue</CardDescription>
                <CardTitle className="text-2xl">$45,231.89</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center text-sm text-green-600">
                  <TrendingUp className="mr-1 h-4 w-4" />
                  +20.1% from last month
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Active Users</CardDescription>
                <CardTitle className="text-2xl">+2,350</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center text-sm text-green-600">
                  <TrendingUp className="mr-1 h-4 w-4" />
                  +180.1% from last month
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Sales</CardDescription>
                <CardTitle className="text-2xl">+12,234</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center text-sm text-green-600">
                  <TrendingUp className="mr-1 h-4 w-4" />
                  +19% from last month
                </div>
              </CardContent>
            </Card>
          </div>
        </ComponentPreview>
      </section>

      {/* Profile Cards Section */}
      <section id="profile" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Profile Cards</h2>
        <p className="mb-6 text-muted-foreground">
          Display user information and team members.
        </p>

        <ComponentPreview title="User Profiles">
          <div className="grid gap-4 md:grid-cols-3">
            <Card className="text-center">
              <CardContent className="pt-6">
                <Avatar className="mx-auto h-16 w-16 mb-4">
                  <AvatarFallback className="text-lg">JD</AvatarFallback>
                </Avatar>
                <h3 className="font-semibold">John Doe</h3>
                <p className="text-sm text-muted-foreground">Product Designer</p>
                <div className="mt-4 flex justify-center gap-2">
                  <Button size="sm">Follow</Button>
                  <Button size="sm" variant="outline">Message</Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="flex items-center gap-4 pt-6">
                <Avatar className="h-12 w-12">
                  <AvatarFallback>SC</AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="font-semibold">Sarah Chen</h3>
                  <p className="text-sm text-muted-foreground">Engineering Lead</p>
                  <div className="mt-1 flex items-center text-xs text-muted-foreground">
                    <MapPin className="mr-1 h-3 w-3" />
                    San Francisco, CA
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <Avatar className="h-10 w-10">
                    <AvatarFallback>MK</AvatarFallback>
                  </Avatar>
                  <Button size="icon" variant="ghost" className="h-8 w-8">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <h3 className="font-semibold">Mike Kim</h3>
                <p className="text-sm text-muted-foreground">Developer</p>
                <div className="mt-3 flex gap-4 text-sm">
                  <div>
                    <span className="font-semibold">142</span>
                    <span className="text-muted-foreground ml-1">Posts</span>
                  </div>
                  <div>
                    <span className="font-semibold">2.4k</span>
                    <span className="text-muted-foreground ml-1">Followers</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </ComponentPreview>
      </section>

      {/* Pricing Cards Section */}
      <section id="pricing" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Pricing Cards</h2>
        <p className="mb-6 text-muted-foreground">
          Showcase pricing plans and subscription tiers.
        </p>

        <ComponentPreview title="Pricing Tiers">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle>Basic</CardTitle>
                <CardDescription>For individuals</CardDescription>
                <div className="mt-2">
                  <span className="text-3xl font-bold">$9</span>
                  <span className="text-muted-foreground">/month</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {["5 Projects", "10GB Storage", "Email Support"].map((feature) => (
                    <li key={feature} className="flex items-center">
                      <Check className="mr-2 h-4 w-4 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="w-full">Get Started</Button>
              </CardFooter>
            </Card>

            <Card className="border-primary border-2 relative">
              <Badge className="absolute -top-2 left-1/2 -translate-x-1/2">Popular</Badge>
              <CardHeader>
                <CardTitle>Pro</CardTitle>
                <CardDescription>For teams</CardDescription>
                <div className="mt-2">
                  <span className="text-3xl font-bold">$29</span>
                  <span className="text-muted-foreground">/month</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {["Unlimited Projects", "100GB Storage", "Priority Support", "Analytics"].map((feature) => (
                    <li key={feature} className="flex items-center">
                      <Check className="mr-2 h-4 w-4 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button className="w-full">Get Started</Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Enterprise</CardTitle>
                <CardDescription>For organizations</CardDescription>
                <div className="mt-2">
                  <span className="text-3xl font-bold">$99</span>
                  <span className="text-muted-foreground">/month</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {["Everything in Pro", "Unlimited Storage", "24/7 Support", "Custom Integrations"].map((feature) => (
                    <li key={feature} className="flex items-center">
                      <Check className="mr-2 h-4 w-4 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="w-full">Contact Sales</Button>
              </CardFooter>
            </Card>
          </div>
        </ComponentPreview>
      </section>

      {/* Feature Cards Section */}
      <section id="feature" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Feature Cards</h2>
        <p className="mb-6 text-muted-foreground">
          Highlight product features and capabilities.
        </p>

        <ComponentPreview title="Features Grid">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 mb-2">
                  <Zap className="h-5 w-5 text-primary" />
                </div>
                <CardTitle className="text-lg">Lightning Fast</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Optimized performance for the best user experience.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 mb-2">
                  <Shield className="h-5 w-5 text-primary" />
                </div>
                <CardTitle className="text-lg">Secure by Default</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Enterprise-grade security built into every feature.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 mb-2">
                  <Layers className="h-5 w-5 text-primary" />
                </div>
                <CardTitle className="text-lg">Scalable</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Grows with your business from startup to enterprise.
                </p>
              </CardContent>
            </Card>
          </div>
        </ComponentPreview>

        <div className="mt-6">
          <ComponentPreview title="Primary Background">
            <div className="grid gap-4 md:grid-cols-3">
              {[
                { icon: Zap, title: "Fast", desc: "Optimized for speed" },
                { icon: Shield, title: "Secure", desc: "Enterprise security" },
                { icon: Layers, title: "Scalable", desc: "Grows with you" },
              ].map((item) => (
                <Card key={item.title} className="bg-white/10 border-white/20 text-white">
                  <CardHeader>
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/20 mb-2">
                      <item.icon className="h-5 w-5" />
                    </div>
                    <CardTitle className="text-lg text-white">{item.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-white/80">{item.desc}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </ComponentPreview>
        </div>
      </section>

      {/* Product Cards Section */}
      <section id="product" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Product Cards</h2>
        <p className="mb-6 text-muted-foreground">
          E-commerce style product display cards.
        </p>

        <ComponentPreview title="Product Grid">
          <div className="grid gap-4 md:grid-cols-3">
            <Card className="overflow-hidden">
              <div className="aspect-square bg-muted flex items-center justify-center">
                <div className="text-4xl">📦</div>
              </div>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="secondary">New</Badge>
                  <Button size="icon" variant="ghost" className="h-8 w-8">
                    <Heart className="h-4 w-4" />
                  </Button>
                </div>
                <h3 className="font-semibold">Product Name</h3>
                <p className="text-sm text-muted-foreground">Short description</p>
                <div className="mt-2 flex items-center justify-between">
                  <span className="font-bold">$49.99</span>
                  <div className="flex items-center text-sm">
                    <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                    <span className="ml-1">4.8</span>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button className="w-full">Add to Cart</Button>
              </CardFooter>
            </Card>

            <Card className="overflow-hidden">
              <div className="aspect-square bg-muted flex items-center justify-center relative">
                <Badge className="absolute top-2 left-2" variant="destructive">Sale</Badge>
                <div className="text-4xl">🎧</div>
              </div>
              <CardContent className="pt-4">
                <h3 className="font-semibold">Wireless Headphones</h3>
                <p className="text-sm text-muted-foreground">Premium audio quality</p>
                <div className="mt-2 flex items-center gap-2">
                  <span className="font-bold">$79.99</span>
                  <span className="text-sm text-muted-foreground line-through">$99.99</span>
                </div>
              </CardContent>
              <CardFooter>
                <Button className="w-full">Add to Cart</Button>
              </CardFooter>
            </Card>

            <Card className="overflow-hidden">
              <div className="aspect-square bg-muted flex items-center justify-center">
                <div className="text-4xl">⌚</div>
              </div>
              <CardContent className="pt-4">
                <h3 className="font-semibold">Smart Watch</h3>
                <p className="text-sm text-muted-foreground">Track your fitness</p>
                <div className="mt-2 flex items-center justify-between">
                  <span className="font-bold">$199.99</span>
                  <div className="flex items-center text-sm">
                    <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                    <span className="ml-1">4.9</span>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button className="w-full">Add to Cart</Button>
              </CardFooter>
            </Card>
          </div>
        </ComponentPreview>
      </section>
    </ThreeColumnLayout>
  );
}
