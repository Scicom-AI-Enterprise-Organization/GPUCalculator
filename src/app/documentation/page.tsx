"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Link from "next/link";
import {
  Palette,
  Type,
  Component,
  FileCode,
  Layers,
  Settings,
  Download,
  Figma,
  Code2,
  Rocket,
  LayoutGrid,
  PanelLeft,
  FileType,
  FolderOpen,
} from "lucide-react";
import { motion } from "framer-motion";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { fadeIn, staggerContainer, defaultViewport } from "@/lib/animations";

const documentationSidebarSections = [
  {
    items: [
      { label: "Quick Start", href: "#quick-start", icon: Rocket },
      { label: "Colors", href: "#colors", icon: Palette },
      { label: "Typography", href: "#typography", icon: Type },
      { label: "Components", href: "#components", icon: LayoutGrid },
      { label: "Layout", href: "#layout", icon: PanelLeft },
      { label: "CSS Variables", href: "#css-variables", icon: Code2 },
      { label: "Typography Classes", href: "#typography-classes", icon: FileType },
      { label: "Download Tokens", href: "#download-tokens", icon: Download },
      { label: "Resources", href: "#resources", icon: FolderOpen },
    ],
  },
];

const documentationSections = [
  {
    icon: Palette,
    title: "Colors",
    description:
      "Learn how to use the color system including primary, semantic, and palette colors.",
    items: [
      "Primary palette (50-950)",
      "Gray palette",
      "Semantic colors (danger, success, warning)",
      "Dark mode support",
    ],
  },
  {
    icon: Type,
    title: "Typography",
    description: "Typography scale and font usage guidelines.",
    items: [
      "Font families (Inter, Poppins)",
      "Heading styles (XL, LG, MD, SM)",
      "Body text styles",
      "Utility classes",
    ],
  },
  {
    icon: Component,
    title: "Components",
    description: "Pre-built UI components powered by shadcn/ui.",
    items: [
      "Buttons and badges",
      "Cards and dialogs",
      "Form elements",
      "Data display components",
    ],
  },
  {
    icon: Layers,
    title: "Layout",
    description: "Layout patterns and spacing guidelines.",
    items: [
      "Container widths",
      "Spacing scale",
      "Grid system",
      "Responsive breakpoints",
    ],
  },
];

const resources = [
  { icon: Settings, title: "shadcn/ui", desc: "Component library" },
  { icon: Layers, title: "Tailwind CSS", desc: "Utility framework" },
  { icon: FileCode, title: "Next.js", desc: "React framework" },
];

export default function DocumentationPage() {
  return (
    <TwoColumnLayout
      title="Documentation"
      sidebarSections={documentationSidebarSections}
    >
      {/* Quick Start */}
      <section id="quick-start" className="mb-12">
        <motion.div initial="hidden" animate="visible" variants={fadeIn}>
          <Card className="border-primary/30 bg-primary/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileCode className="h-5 w-5" />
                Quick Start
              </CardTitle>
              <CardDescription>
                Get up and running with the design system
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                {
                  step: "1",
                  title: "Import the CSS",
                  code: 'import "./globals.css";',
                },
                {
                  step: "2",
                  title: "Use Tailwind classes",
                  code: '<div className="bg-primary text-primary-foreground">',
                },
                {
                  step: "3",
                  title: "Import components",
                  code: 'import { Button } from "@/components/ui/button";',
                },
              ].map((item, index) => (
                <motion.div
                  key={item.step}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <h4 className="mb-2 font-medium">
                    {item.step}. {item.title}
                  </h4>
                  <code className="block rounded-lg bg-muted p-4 text-sm">
                    {item.code}
                  </code>
                </motion.div>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </section>

      {/* Documentation Sections */}
      <section id="documentation-sections" className="mb-12">
        <h2 className="mb-6 text-2xl font-semibold">Documentation Sections</h2>
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={defaultViewport}
          variants={staggerContainer}
          className="grid gap-6 md:grid-cols-2"
        >
          {documentationSections.map((section, index) => (
            <motion.div
              key={section.title}
              id={section.title.toLowerCase()}
              variants={fadeIn}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -4 }}
            >
              <Card className="h-full transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <section.icon className="h-5 w-5 text-primary" />
                  </div>
                  <CardTitle>{section.title}</CardTitle>
                  <CardDescription>{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {section.items.map((item) => (
                      <li
                        key={item}
                        className="flex items-center gap-2 text-sm text-muted-foreground"
                      >
                        <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* CSS Variables Reference */}
      <section id="css-variables" className="mb-12">
        <h2 className="mb-6 text-2xl font-semibold">CSS Variables Reference</h2>
        <Card>
          <CardHeader>
            <CardTitle>Color Variables</CardTitle>
            <CardDescription>
              Available CSS custom properties for colors
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-2">
                <h4 className="font-medium">Semantic</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>
                    <code>--background</code>
                  </li>
                  <li>
                    <code>--foreground</code>
                  </li>
                  <li>
                    <code>--primary</code>
                  </li>
                  <li>
                    <code>--secondary</code>
                  </li>
                  <li>
                    <code>--muted</code>
                  </li>
                  <li>
                    <code>--accent</code>
                  </li>
                  <li>
                    <code>--destructive</code>
                  </li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">Primary Palette</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>
                    <code>--primary-50</code> to <code>--primary-950</code>
                  </li>
                </ul>
                <h4 className="mt-4 font-medium">Gray Palette</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>
                    <code>--gray-50</code> to <code>--gray-950</code>
                  </li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">Status Colors</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>
                    <code>--danger-50</code> to <code>--danger-950</code>
                  </li>
                  <li>
                    <code>--success-50</code> to <code>--success-950</code>
                  </li>
                  <li>
                    <code>--warning-50</code> to <code>--warning-950</code>
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Typography Classes */}
      <section id="typography-classes" className="mb-12">
        <h2 className="mb-6 text-2xl font-semibold">Typography Classes</h2>
        <Card>
          <CardHeader>
            <CardTitle>Heading Utilities</CardTitle>
            <CardDescription>
              Custom typography classes from your Figma design
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { class: ".heading-xl", desc: "60px, Poppins, 600 weight" },
                { class: ".heading-lg", desc: "48px, Inter, 600 weight" },
                { class: ".heading-md", desc: "36px, Inter, 600 weight" },
                { class: ".heading-sm", desc: "30px, Inter, 500 weight" },
              ].map((item, index) => (
                <motion.div
                  key={item.class}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between border-b border-border pb-4 last:border-0"
                >
                  <div>
                    <code className="text-sm">{item.class}</code>
                    <p className="text-sm text-muted-foreground">{item.desc}</p>
                  </div>
                  <span className={item.class.replace(".", "")}>Aa</span>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Download Tokens */}
      <section id="download-tokens" className="mb-12">
        <h2 className="mb-6 text-2xl font-semibold">Download Design Tokens</h2>
        <p className="mb-6 text-muted-foreground">
          Export design tokens for use in Figma or your development environment.
        </p>
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={defaultViewport}
          variants={staggerContainer}
          className="grid gap-6 md:grid-cols-2"
        >
          {/* Figma / Tokens Studio */}
          <motion.div variants={fadeIn} whileHover={{ y: -4 }}>
            <Card className="h-full border-primary/30 bg-primary/10">
              <CardHeader>
                <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                  <Figma className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>For Designers (Figma)</CardTitle>
                <CardDescription>
                  JSON tokens compatible with Tokens Studio plugin
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    All color palettes (Primary, Gray, Status)
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Typography tokens (fonts, sizes, weights)
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Spacing and border radius scales
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Multi-brand themes (Scicom, EMGS, Telekom)
                  </li>
                </ul>
                <Button asChild className="w-full">
                  <a
                    href="/tokens/design-tokens.json"
                    download="design-tokens.json"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download JSON Tokens
                  </a>
                </Button>
                <p className="text-xs text-muted-foreground">
                  Import into Figma using{" "}
                  <Link
                    href="https://tokens.studio"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Tokens Studio
                  </Link>{" "}
                  plugin
                </p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Developers */}
          <motion.div variants={fadeIn} whileHover={{ y: -4 }}>
            <Card className="h-full">
              <CardHeader>
                <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                  <Code2 className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>For Developers</CardTitle>
                <CardDescription>
                  CSS variables file ready for any project
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Pure CSS custom properties
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    No build tools required
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Theme classes for brand switching
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Dark mode support included
                  </li>
                </ul>
                <Button asChild variant="outline" className="w-full">
                  <a href="/tokens/variables.css" download="variables.css">
                    <Download className="mr-2 h-4 w-4" />
                    Download CSS Variables
                  </a>
                </Button>
                <p className="text-xs text-muted-foreground">
                  Import with:{" "}
                  <code className="rounded bg-muted px-1 py-0.5">
                    @import &apos;variables.css&apos;;
                  </code>
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </section>

      {/* Resources */}
      <section id="resources" className="mb-0">
        <h2 className="mb-6 text-2xl font-semibold">Resources</h2>
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={defaultViewport}
          variants={staggerContainer}
          className="grid gap-4 md:grid-cols-3"
        >
          {resources.map((item, index) => (
            <motion.div
              key={item.title}
              variants={fadeIn}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -4 }}
            >
              <Card className="transition-shadow hover:shadow-lg">
                <CardContent className="flex items-center gap-4 pt-6">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                    <item.icon className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold">{item.title}</h3>
                    <p className="text-sm text-muted-foreground">{item.desc}</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </section>
    </TwoColumnLayout>
  );
}
