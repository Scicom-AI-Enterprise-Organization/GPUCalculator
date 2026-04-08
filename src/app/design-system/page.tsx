"use client";

import { useMemo, useCallback } from "react";
import Link from "next/link";
import { useBrand } from "@/components/brand-provider";
import {
  brandTokenConfigs,
  generateCSS,
  generateJSON,
  downloadFile,
} from "@/lib/brand-tokens";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { fadeIn, staggerContainer, defaultViewport } from "@/lib/animations";
import {
  Download,
  Figma,
  Code2,
  Palette,
  Paintbrush,
  Type,
  Shapes,
  Square,
  Layers,
  Ruler,
  Home,
  Search,
  Settings,
  User,
  Bell,
  Mail,
  Calendar,
  FileText,
  BarChart3,
  ArrowRight,
  ArrowLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Check,
  X,
  Plus,
  Minus,
  Edit,
  Trash2,
  Eye,
  EyeOff,
  Upload,
  Filter,
  RefreshCw,
  ExternalLink,
  Copy,
  MoreHorizontal,
  MoreVertical,
} from "lucide-react";

// Brand-specific font configurations
const brandFontConfigs = {
  scicom: {
    displayFont: "Satoshi",
    bodyFont: "Inter",
    displayWeight: "Semibold (600)",
    headingWeight: "Semibold (600)",
    bodyWeight: "Regular (400)",
    displayFontFamily: "var(--font-display), Satoshi, sans-serif",
    bodyFontFamily: "var(--font-inter), Inter, sans-serif",
    displayFontUrl: "https://www.fontshare.com/fonts/satoshi",
    bodyFontUrl: "https://fonts.google.com/specimen/Inter",
  },
  emgs: {
    displayFont: "Montserrat",
    bodyFont: "Montserrat",
    displayWeight: "Black (900)",
    headingWeight: "Black (900)",
    bodyWeight: "Bold (700)",
    displayFontFamily: "var(--font-montserrat), Montserrat, sans-serif",
    bodyFontFamily: "var(--font-montserrat), Montserrat, sans-serif",
    displayFontUrl: "https://fonts.google.com/specimen/Montserrat",
    bodyFontUrl: "https://fonts.google.com/specimen/Montserrat",
  },
  telekom: {
    displayFont: "Open Sans",
    bodyFont: "Open Sans",
    displayWeight: "Semibold (600)",
    headingWeight: "Semibold (600)",
    bodyWeight: "Regular (400)",
    displayFontFamily: "var(--font-open-sans), Open Sans, sans-serif",
    bodyFontFamily: "var(--font-open-sans), Open Sans, sans-serif",
    displayFontUrl: "https://fonts.google.com/specimen/Open+Sans",
    bodyFontUrl: "https://fonts.google.com/specimen/Open+Sans",
  },
};

// Brand-specific color palettes
const brandColorPalettes = {
  scicom: {
    Primary: [
      { name: "50", hex: "#f2f5fc" },
      { name: "100", hex: "#dee8fd" },
      { name: "200", hex: "#c2d4fd" },
      { name: "300", hex: "#90b6ff" },
      { name: "400", hex: "#6193ff" },
      { name: "500", hex: "#3a81f6" },
      { name: "600", hex: "#2563ef" },
      { name: "700", hex: "#284cc3" },
      { name: "800", hex: "#1f3fad" },
      { name: "900", hex: "#1a2c77" },
      { name: "950", hex: "#162455" },
    ],
    Secondary: [
      { name: "50", hex: "#fff1eb" },
      { name: "100", hex: "#ffdfd5" },
      { name: "200", hex: "#fdd3c9" },
      { name: "300", hex: "#ffbaa4" },
      { name: "400", hex: "#ffa381" },
      { name: "500", hex: "#ff8950" },
      { name: "600", hex: "#f87100" },
      { name: "700", hex: "#b75301" },
      { name: "800", hex: "#7b3604" },
      { name: "900", hex: "#471900" },
      { name: "950", hex: "#2d0d00" },
    ],
  },
  emgs: {
    Primary: [
      { name: "50", hex: "#fdedee" },
      { name: "100", hex: "#fadbde" },
      { name: "200", hex: "#f7babf" },
      { name: "300", hex: "#f3929d" },
      { name: "400", hex: "#f16a7c" },
      { name: "500", hex: "#e73457" },
      { name: "600", hex: "#be2946" },
      { name: "700", hex: "#911d33" },
      { name: "800", hex: "#641121" },
      { name: "900", hex: "#3f0712" },
      { name: "950", hex: "#290309" },
    ],
    Secondary: [
      { name: "50", hex: "#fff5ee" },
      { name: "100", hex: "#feeee3" },
      { name: "200", hex: "#fed9be" },
      { name: "300", hex: "#fec79a" },
      { name: "400", hex: "#fdb160" },
      { name: "500", hex: "#f6a020" },
      { name: "600", hex: "#c17d17" },
      { name: "700", hex: "#915d0e" },
      { name: "800", hex: "#613c06" },
      { name: "900", hex: "#382102" },
      { name: "950", hex: "#221201" },
    ],
  },
  telekom: {
    Primary: [
      { name: "50", hex: "#e6f3ff" },
      { name: "100", hex: "#cce7ff" },
      { name: "200", hex: "#99cfff" },
      { name: "300", hex: "#66b7ff" },
      { name: "400", hex: "#339fff" },
      { name: "500", hex: "#0087ff" },
      { name: "600", hex: "#0066b3" },
      { name: "700", hex: "#005299" },
      { name: "800", hex: "#003d73" },
      { name: "900", hex: "#00294d" },
      { name: "950", hex: "#001426" },
    ],
    Secondary: [
      { name: "50", hex: "#e6f3ff" },
      { name: "100", hex: "#cce7ff" },
      { name: "200", hex: "#99cfff" },
      { name: "300", hex: "#66b7ff" },
      { name: "400", hex: "#339fff" },
      { name: "500", hex: "#00a0e4" },
      { name: "600", hex: "#0080b6" },
      { name: "700", hex: "#006088" },
      { name: "800", hex: "#00405b" },
      { name: "900", hex: "#00202d" },
      { name: "950", hex: "#001017" },
    ],
  },
};

// Heading typography scale data
const headingTypography = [
  { name: "Heading Extra Large", tag: "", fontSize: "60px (3.75rem)", lineHeight: "72px (4.5rem)" },
  { name: "Heading Large", tag: "", fontSize: "48px (3rem)", lineHeight: "60px (3.75rem)" },
  { name: "Heading Medium", tag: "h1", fontSize: "36px (2.25rem)", lineHeight: "44px (2.75rem)" },
  { name: "Heading Small", tag: "h2", fontSize: "30px (1.875rem)", lineHeight: "38px (2.375rem)" },
  { name: "Heading Extra Small", tag: "h3", fontSize: "24px (1.5rem)", lineHeight: "32px (2rem)" },
  { name: "Heading 2X Small", tag: "h4", fontSize: "20px (1.25rem)", lineHeight: "28px (1.75rem)" },
  { name: "Heading 3X Small", tag: "h5", fontSize: "16px (1rem)", lineHeight: "24px (1.5rem)" },
  { name: "Heading 4X Small", tag: "h6", fontSize: "14px (0.875rem)", lineHeight: "20px (1.25rem)" },
];

// Body typography scale data
const bodyTypography = [
  { name: "Body Extra Large", tag: "p", fontSize: "20px (1.25rem)", lineHeight: "30px (1.875rem)" },
  { name: "Body Large", tag: "p", fontSize: "18px (1.125rem)", lineHeight: "28px (1.75rem)" },
  { name: "Body Medium", tag: "p", fontSize: "16px (1rem)", lineHeight: "24px (1.5rem)" },
  { name: "Body Small", tag: "p", fontSize: "14px (0.875rem)", lineHeight: "20px (1.25rem)" },
  { name: "Body Extra Small", tag: "p", fontSize: "12px (0.75rem)", lineHeight: "18px (1.125rem)" },
  { name: "Caption", tag: "span", fontSize: "12px (0.75rem)", lineHeight: "16px (1rem)" },
  { name: "Overline", tag: "span", fontSize: "11px (0.6875rem)", lineHeight: "16px (1rem)" },
];

const designSystemSidebarSections = [
  {
    items: [
      { label: "Download Tokens", href: "#download-tokens", icon: Download },
      { label: "Semantic Color Reference", href: "#semantic-colors", icon: Palette },
      { label: "Color Palettes", href: "#color-palettes", icon: Paintbrush },
      { label: "Typography Scale", href: "#typography", icon: Type },
      { label: "Icons", href: "#icons", icon: Shapes },
      { label: "Border Radius", href: "#border-radius", icon: Square },
      { label: "Shadows", href: "#shadows", icon: Layers },
      { label: "Spacing", href: "#spacing", icon: Ruler },
    ],
  },
];

// Complete color palettes from Scicom design system (Preline-based)
const sharedColorPalettes = {
  Red: [
    { name: "50", hex: "#fef2f2" }, { name: "100", hex: "#fee2e2" }, { name: "200", hex: "#fecaca" },
    { name: "300", hex: "#fca5a5" }, { name: "400", hex: "#f87171" }, { name: "500", hex: "#ef4444" },
    { name: "600", hex: "#dc2626" }, { name: "700", hex: "#b91c1c" }, { name: "800", hex: "#991b1b" },
    { name: "900", hex: "#7f1d1d" }, { name: "950", hex: "#450a0a" },
  ],
  Orange: [
    { name: "50", hex: "#fff4f2" }, { name: "100", hex: "#ffe9e4" }, { name: "200", hex: "#ffd3c8" },
    { name: "300", hex: "#ffb9a4" }, { name: "400", hex: "#ffa17f" }, { name: "500", hex: "#ff884f" },
    { name: "600", hex: "#f57100" }, { name: "700", hex: "#b65200" }, { name: "800", hex: "#7b3604" },
    { name: "900", hex: "#461b00" }, { name: "950", hex: "#2d0f00" },
  ],
  Amber: [
    { name: "50", hex: "#fffbeb" }, { name: "100", hex: "#fef3c7" }, { name: "200", hex: "#fde68a" },
    { name: "300", hex: "#fcd34d" }, { name: "400", hex: "#fbbf24" }, { name: "500", hex: "#f59e0b" },
    { name: "600", hex: "#d97706" }, { name: "700", hex: "#b45309" }, { name: "800", hex: "#92400e" },
    { name: "900", hex: "#78350f" }, { name: "950", hex: "#451a03" },
  ],
  Yellow: [
    { name: "50", hex: "#fefce8" }, { name: "100", hex: "#fef9c3" }, { name: "200", hex: "#fef08a" },
    { name: "300", hex: "#fde047" }, { name: "400", hex: "#facc15" }, { name: "500", hex: "#eab308" },
    { name: "600", hex: "#ca8a04" }, { name: "700", hex: "#a16207" }, { name: "800", hex: "#854d0e" },
    { name: "900", hex: "#713f12" }, { name: "950", hex: "#422006" },
  ],
  Lime: [
    { name: "50", hex: "#f7fee7" }, { name: "100", hex: "#ecfccb" }, { name: "200", hex: "#d9f99d" },
    { name: "300", hex: "#bef264" }, { name: "400", hex: "#a3e635" }, { name: "500", hex: "#84cc16" },
    { name: "600", hex: "#65a30d" }, { name: "700", hex: "#4d7c0f" }, { name: "800", hex: "#3f6212" },
    { name: "900", hex: "#365314" }, { name: "950", hex: "#1a2e05" },
  ],
  Green: [
    { name: "50", hex: "#f0fdf4" }, { name: "100", hex: "#dcfce7" }, { name: "200", hex: "#bbf7d0" },
    { name: "300", hex: "#86efac" }, { name: "400", hex: "#4ade80" }, { name: "500", hex: "#22c55e" },
    { name: "600", hex: "#16a34a" }, { name: "700", hex: "#15803d" }, { name: "800", hex: "#166534" },
    { name: "900", hex: "#14532d" }, { name: "950", hex: "#052e16" },
  ],
  Emerald: [
    { name: "50", hex: "#ecfdf5" }, { name: "100", hex: "#d1fae5" }, { name: "200", hex: "#a7f3d0" },
    { name: "300", hex: "#6ee7b7" }, { name: "400", hex: "#34d399" }, { name: "500", hex: "#10b981" },
    { name: "600", hex: "#059669" }, { name: "700", hex: "#047857" }, { name: "800", hex: "#065f46" },
    { name: "900", hex: "#064e3b" }, { name: "950", hex: "#022c22" },
  ],
  Teal: [
    { name: "50", hex: "#f0fdfa" }, { name: "100", hex: "#ccfbf1" }, { name: "200", hex: "#99f6e4" },
    { name: "300", hex: "#5eead4" }, { name: "400", hex: "#2dd4bf" }, { name: "500", hex: "#14b8a6" },
    { name: "600", hex: "#0d9488" }, { name: "700", hex: "#0f766e" }, { name: "800", hex: "#115e59" },
    { name: "900", hex: "#134e4a" }, { name: "950", hex: "#042f2e" },
  ],
  Cyan: [
    { name: "50", hex: "#ecfeff" }, { name: "100", hex: "#cffafe" }, { name: "200", hex: "#a5f3fc" },
    { name: "300", hex: "#67e8f9" }, { name: "400", hex: "#22d3ee" }, { name: "500", hex: "#06b6d4" },
    { name: "600", hex: "#0891b2" }, { name: "700", hex: "#0e7490" }, { name: "800", hex: "#155e75" },
    { name: "900", hex: "#164e63" }, { name: "950", hex: "#083344" },
  ],
  Sky: [
    { name: "50", hex: "#f0f9ff" }, { name: "100", hex: "#e0f2fe" }, { name: "200", hex: "#bae6fd" },
    { name: "300", hex: "#7dd3fc" }, { name: "400", hex: "#38bdf8" }, { name: "500", hex: "#0ea5e9" },
    { name: "600", hex: "#0284c7" }, { name: "700", hex: "#0369a1" }, { name: "800", hex: "#075985" },
    { name: "900", hex: "#0c4a6e" }, { name: "950", hex: "#082f49" },
  ],
  Blue: [
    { name: "50", hex: "#eff6ff" }, { name: "100", hex: "#dbeafe" }, { name: "200", hex: "#bfdbfe" },
    { name: "300", hex: "#93c5fd" }, { name: "400", hex: "#60a5fa" }, { name: "500", hex: "#3b82f6" },
    { name: "600", hex: "#2563eb" }, { name: "700", hex: "#1d4ed8" }, { name: "800", hex: "#1e40af" },
    { name: "900", hex: "#1e3a8a" }, { name: "950", hex: "#172554" },
  ],
  Indigo: [
    { name: "50", hex: "#eef2ff" }, { name: "100", hex: "#e0e7ff" }, { name: "200", hex: "#c7d2fe" },
    { name: "300", hex: "#a5b4fc" }, { name: "400", hex: "#818cf8" }, { name: "500", hex: "#6366f1" },
    { name: "600", hex: "#4f46e5" }, { name: "700", hex: "#4338ca" }, { name: "800", hex: "#3730a3" },
    { name: "900", hex: "#312e81" }, { name: "950", hex: "#1e1b4b" },
  ],
  Violet: [
    { name: "50", hex: "#f5f3ff" }, { name: "100", hex: "#ede9fe" }, { name: "200", hex: "#ddd6fe" },
    { name: "300", hex: "#c4b5fd" }, { name: "400", hex: "#a78bfa" }, { name: "500", hex: "#8b5cf6" },
    { name: "600", hex: "#7c3aed" }, { name: "700", hex: "#6d28d9" }, { name: "800", hex: "#5b21b6" },
    { name: "900", hex: "#4c1d95" }, { name: "950", hex: "#2e1065" },
  ],
  Purple: [
    { name: "50", hex: "#faf5ff" }, { name: "100", hex: "#f3e8ff" }, { name: "200", hex: "#e9d5ff" },
    { name: "300", hex: "#d8b4fe" }, { name: "400", hex: "#c084fc" }, { name: "500", hex: "#a855f7" },
    { name: "600", hex: "#9333ea" }, { name: "700", hex: "#7e22ce" }, { name: "800", hex: "#6b21a8" },
    { name: "900", hex: "#581c87" }, { name: "950", hex: "#3b0764" },
  ],
  Fuchsia: [
    { name: "50", hex: "#fdf4ff" }, { name: "100", hex: "#fae8ff" }, { name: "200", hex: "#f5d0fe" },
    { name: "300", hex: "#f0abfc" }, { name: "400", hex: "#e879f9" }, { name: "500", hex: "#d946ef" },
    { name: "600", hex: "#c026d3" }, { name: "700", hex: "#a21caf" }, { name: "800", hex: "#86198f" },
    { name: "900", hex: "#701a75" }, { name: "950", hex: "#4a044e" },
  ],
  Pink: [
    { name: "50", hex: "#fdf2f8" }, { name: "100", hex: "#fce7f3" }, { name: "200", hex: "#fbcfe8" },
    { name: "300", hex: "#f9a8d4" }, { name: "400", hex: "#f472b6" }, { name: "500", hex: "#ec4899" },
    { name: "600", hex: "#db2777" }, { name: "700", hex: "#be185d" }, { name: "800", hex: "#9d174d" },
    { name: "900", hex: "#831843" }, { name: "950", hex: "#500724" },
  ],
  Rose: [
    { name: "50", hex: "#fff1f2" }, { name: "100", hex: "#ffe4e6" }, { name: "200", hex: "#fecdd3" },
    { name: "300", hex: "#fda4af" }, { name: "400", hex: "#fb7185" }, { name: "500", hex: "#f43f5e" },
    { name: "600", hex: "#e11d48" }, { name: "700", hex: "#be123c" }, { name: "800", hex: "#9f1239" },
    { name: "900", hex: "#881337" }, { name: "950", hex: "#4c0519" },
  ],
  Slate: [
    { name: "50", hex: "#f8fafc" }, { name: "100", hex: "#f1f5f9" }, { name: "200", hex: "#e2e8f0" },
    { name: "300", hex: "#cbd5e1" }, { name: "400", hex: "#94a3b8" }, { name: "500", hex: "#64748b" },
    { name: "600", hex: "#475569" }, { name: "700", hex: "#334155" }, { name: "800", hex: "#1e293b" },
    { name: "900", hex: "#0f172a" }, { name: "950", hex: "#020617" },
  ],
  Gray: [
    { name: "50", hex: "#f9fafb" }, { name: "100", hex: "#f3f4f6" }, { name: "200", hex: "#e5e7eb" },
    { name: "300", hex: "#d1d5db" }, { name: "400", hex: "#9ca3af" }, { name: "500", hex: "#6b7280" },
    { name: "600", hex: "#4b5563" }, { name: "700", hex: "#374151" }, { name: "800", hex: "#1f2937" },
    { name: "900", hex: "#111827" }, { name: "950", hex: "#030712" },
  ],
  Zinc: [
    { name: "50", hex: "#fafafa" }, { name: "100", hex: "#f4f4f5" }, { name: "200", hex: "#e4e4e7" },
    { name: "300", hex: "#d4d4d8" }, { name: "400", hex: "#a1a1aa" }, { name: "500", hex: "#71717a" },
    { name: "600", hex: "#52525b" }, { name: "700", hex: "#3f3f46" }, { name: "800", hex: "#27272a" },
    { name: "900", hex: "#18181b" }, { name: "950", hex: "#09090b" },
  ],
  Neutral: [
    { name: "50", hex: "#fafafa" }, { name: "100", hex: "#f5f5f5" }, { name: "200", hex: "#e5e5e5" },
    { name: "300", hex: "#d4d4d4" }, { name: "400", hex: "#a3a3a3" }, { name: "500", hex: "#737373" },
    { name: "600", hex: "#525252" }, { name: "700", hex: "#404040" }, { name: "800", hex: "#262626" },
    { name: "900", hex: "#171717" }, { name: "950", hex: "#0a0a0a" },
  ],
  Stone: [
    { name: "50", hex: "#fafaf9" }, { name: "100", hex: "#f5f5f4" }, { name: "200", hex: "#e7e5e4" },
    { name: "300", hex: "#d6d3d1" }, { name: "400", hex: "#a8a29e" }, { name: "500", hex: "#78716c" },
    { name: "600", hex: "#57534e" }, { name: "700", hex: "#44403c" }, { name: "800", hex: "#292524" },
    { name: "900", hex: "#1c1917" }, { name: "950", hex: "#0c0a09" },
  ],
  Taupe: [
    { name: "50", hex: "#fbfaf9" }, { name: "100", hex: "#f3f1f1" }, { name: "200", hex: "#e8e4e3" },
    { name: "300", hex: "#d7d0d7" }, { name: "400", hex: "#aba09c" }, { name: "500", hex: "#7c6d67" },
    { name: "600", hex: "#5b4f4b" }, { name: "700", hex: "#473c39" }, { name: "800", hex: "#2b2422" },
    { name: "900", hex: "#1d1816" }, { name: "950", hex: "#0c0a09" },
  ],
  Mauve: [
    { name: "50", hex: "#fafafa" }, { name: "100", hex: "#f3f1f3" }, { name: "200", hex: "#e7e4e7" },
    { name: "300", hex: "#d7d0d7" }, { name: "400", hex: "#a89ea9" }, { name: "500", hex: "#79697b" },
    { name: "600", hex: "#594c5b" }, { name: "700", hex: "#463947" }, { name: "800", hex: "#2a212c" },
    { name: "900", hex: "#1d161e" }, { name: "950", hex: "#0c090c" },
  ],
  Mist: [
    { name: "50", hex: "#f9fbfb" }, { name: "100", hex: "#f1f3f3" }, { name: "200", hex: "#e3e7e8" },
    { name: "300", hex: "#d0d6d8" }, { name: "400", hex: "#9ca8ab" }, { name: "500", hex: "#67787c" },
    { name: "600", hex: "#4b585b" }, { name: "700", hex: "#394447" }, { name: "800", hex: "#22292b" },
    { name: "900", hex: "#161b1d" }, { name: "950", hex: "#090b0c" },
  ],
  Olive: [
    { name: "50", hex: "#fbfbf9" }, { name: "100", hex: "#f4f4f0" }, { name: "200", hex: "#e8e8e3" },
    { name: "300", hex: "#d8d8d0" }, { name: "400", hex: "#abab9c" }, { name: "500", hex: "#7c7c67" },
    { name: "600", hex: "#5b5b4b" }, { name: "700", hex: "#474739" }, { name: "800", hex: "#2b2b22" },
    { name: "900", hex: "#1d1d16" }, { name: "950", hex: "#0c0c09" },
  ],
  "Preline-Khaki": [
    { name: "50", hex: "#f9f8f7" }, { name: "100", hex: "#f1f0ed" }, { name: "200", hex: "#dfddd7" },
    { name: "300", hex: "#cac6bc" }, { name: "400", hex: "#aea797" }, { name: "500", hex: "#8f8773" },
    { name: "600", hex: "#726b57" }, { name: "700", hex: "#514b3b" }, { name: "800", hex: "#3e392c" },
    { name: "900", hex: "#332f25" }, { name: "950", hex: "#1a160a" },
  ],
  "Preline-Mauve": [
    { name: "50", hex: "#faf8f9" }, { name: "100", hex: "#f3eff2" }, { name: "200", hex: "#e3dae0" },
    { name: "300", hex: "#d0c0ca" }, { name: "400", hex: "#b59bab" }, { name: "500", hex: "#876a7c" },
    { name: "600", hex: "#7a5c6e" }, { name: "700", hex: "#674e5d" }, { name: "800", hex: "#4b3743" },
    { name: "900", hex: "#3e2e38" }, { name: "950", hex: "#25141f" },
  ],
  "Preline-Avocado": [
    { name: "50", hex: "#e6e8e0" }, { name: "100", hex: "#d7dccd" }, { name: "200", hex: "#bcc3aa" },
    { name: "300", hex: "#a1ab85" }, { name: "400", hex: "#889560" }, { name: "500", hex: "#717f43" },
    { name: "600", hex: "#5c6932" }, { name: "700", hex: "#495425" }, { name: "800", hex: "#38401d" },
    { name: "900", hex: "#282d16" }, { name: "950", hex: "#202514" },
  ],
  "Preline-Avocado-Soft": [
    { name: "50", hex: "#f8f9f7" }, { name: "100", hex: "#f0f0ee" }, { name: "200", hex: "#dcded8" },
    { name: "300", hex: "#c5c8bf" }, { name: "400", hex: "#a6aa9d" }, { name: "500", hex: "#868a7c" },
    { name: "600", hex: "#6a6d63" }, { name: "700", hex: "#4a4c45" }, { name: "800", hex: "#262723" },
    { name: "900", hex: "#171715" }, { name: "950", hex: "#090909" },
  ],
};

// Semantic color reference — grouped by purpose, with light & dark hex values
const semanticColorGroups = [
  {
    title: "Primary Theme Colors",
    colors: [
      { name: "Background", variable: "--background", tailwind: "bg-background", light: "#f3f4f6", dark: "#171717" },
      { name: "Foreground", variable: "--foreground", tailwind: "text-foreground", light: "#18181b", dark: "#e5e5e5" },
      { name: "Primary", variable: "--primary", tailwind: "bg-primary", light: "#3b82f6", dark: "#f57100" },
      { name: "Primary Foreground", variable: "--primary-foreground", tailwind: "text-primary-foreground", light: "#f9fafb", dark: "#ffffff" },
    ],
  },
  {
    title: "Secondary & Accent Colors",
    colors: [
      { name: "Secondary", variable: "--secondary", tailwind: "bg-secondary", light: "#2563eb", dark: "#ff884f" },
      { name: "Secondary Foreground", variable: "--secondary-foreground", tailwind: "text-secondary-foreground", light: "#f9fafb", dark: "#ffffff" },
      { name: "Accent", variable: "--accent", tailwind: "bg-accent", light: "#e5e7eb", dark: "#525252" },
      { name: "Accent Foreground", variable: "--accent-foreground", tailwind: "text-accent-foreground", light: "#18181b", dark: "#e5e5e5" },
    ],
  },
  {
    title: "UI Component Colors",
    colors: [
      { name: "Card", variable: "--card", tailwind: "bg-card", light: "#f9fafb", dark: "#262626" },
      { name: "Card Foreground", variable: "--card-foreground", tailwind: "text-card-foreground", light: "#333333", dark: "#e5e5e5" },
      { name: "Popover", variable: "--popover", tailwind: "bg-popover", light: "#ffffff", dark: "#262626" },
      { name: "Popover Foreground", variable: "--popover-foreground", tailwind: "text-popover-foreground", light: "#333333", dark: "#e5e5e5" },
      { name: "Muted", variable: "--muted", tailwind: "bg-muted", light: "#f9fafb", dark: "#1f1f1f" },
      { name: "Muted Foreground", variable: "--muted-foreground", tailwind: "text-muted-foreground", light: "#6b7280", dark: "#a3a3a3" },
    ],
  },
  {
    title: "Utility & Form Colors",
    colors: [
      { name: "Border", variable: "--border", tailwind: "border-border", light: "#e5e7eb", dark: "#404040" },
      { name: "Input", variable: "--input", tailwind: "border-input", light: "#e5e7eb", dark: "#525252" },
      { name: "Ring", variable: "--ring", tailwind: "ring-ring", light: "#2563eb", dark: "#ff884f" },
    ],
  },
  {
    title: "Status & Feedback Colors",
    colors: [
      { name: "Destructive", variable: "--destructive", tailwind: "bg-destructive", light: "#ef4444", dark: "#f87171" },
      { name: "Destructive Foreground", variable: "--destructive-foreground", tailwind: "text-destructive-foreground", light: "#ffffff", dark: "#ffffff" },
    ],
  },
  {
    title: "Chart & Visualization Colors",
    colors: [
      { name: "Chart 1", variable: "--chart-1", tailwind: "text-chart-1", light: "#c2d5ff", dark: "#ffb9a4" },
      { name: "Chart 2", variable: "--chart-2", tailwind: "text-chart-2", light: "#6394ff", dark: "#ff884f" },
      { name: "Chart 3", variable: "--chart-3", tailwind: "text-chart-3", light: "#2563eb", dark: "#b65200" },
      { name: "Chart 4", variable: "--chart-4", tailwind: "text-chart-4", light: "#1e40af", dark: "#461b00" },
      { name: "Chart 5", variable: "--chart-5", tailwind: "text-chart-5", light: "#172554", dark: "#2d0f00" },
    ],
  },
  {
    title: "Sidebar & Navigation Colors",
    colors: [
      { name: "Sidebar", variable: "--sidebar", tailwind: "bg-sidebar", light: "#f9fafb", dark: "#262626" },
      { name: "Sidebar Foreground", variable: "--sidebar-foreground", tailwind: "text-sidebar-foreground", light: "#333333", dark: "#e5e5e5" },
      { name: "Sidebar Primary", variable: "--sidebar-primary", tailwind: "bg-sidebar-primary", light: "#3b82f6", dark: "#ff884f" },
      { name: "Sidebar Primary FG", variable: "--sidebar-primary-foreground", tailwind: "text-sidebar-primary-foreground", light: "#ffffff", dark: "#ffffff" },
      { name: "Sidebar Accent", variable: "--sidebar-accent", tailwind: "bg-sidebar-accent", light: "#e5e7eb", dark: "#404040" },
      { name: "Sidebar Accent FG", variable: "--sidebar-accent-foreground", tailwind: "text-sidebar-accent-foreground", light: "#1e40af", dark: "#fff4f2" },
      { name: "Sidebar Border", variable: "--sidebar-border", tailwind: "border-sidebar-border", light: "#e5e7eb", dark: "#404040" },
      { name: "Sidebar Ring", variable: "--sidebar-ring", tailwind: "ring-sidebar-ring", light: "#2563eb", dark: "#ffd3c8" },
    ],
  },
];

// Icons data
const iconList = [
  { icon: Home, name: "Home" },
  { icon: Search, name: "Search" },
  { icon: Settings, name: "Settings" },
  { icon: User, name: "User" },
  { icon: Bell, name: "Bell" },
  { icon: Mail, name: "Mail" },
  { icon: Calendar, name: "Calendar" },
  { icon: FileText, name: "FileText" },
  { icon: BarChart3, name: "BarChart3" },
  { icon: ArrowRight, name: "ArrowRight" },
  { icon: ArrowLeft, name: "ArrowLeft" },
  { icon: ChevronRight, name: "ChevronRight" },
  { icon: ChevronDown, name: "ChevronDown" },
  { icon: ChevronUp, name: "ChevronUp" },
  { icon: Check, name: "Check" },
  { icon: X, name: "X" },
  { icon: Plus, name: "Plus" },
  { icon: Minus, name: "Minus" },
  { icon: Edit, name: "Edit" },
  { icon: Trash2, name: "Trash2" },
  { icon: Eye, name: "Eye" },
  { icon: EyeOff, name: "EyeOff" },
  { icon: Download, name: "Download" },
  { icon: Upload, name: "Upload" },
  { icon: Filter, name: "Filter" },
  { icon: RefreshCw, name: "RefreshCw" },
  { icon: ExternalLink, name: "ExternalLink" },
  { icon: Copy, name: "Copy" },
  { icon: MoreHorizontal, name: "MoreHorizontal" },
  { icon: MoreVertical, name: "MoreVertical" },
];

const iconSizes = [
  { label: "xs", size: 12 },
  { label: "sm", size: 16 },
  { label: "md", size: 20 },
  { label: "lg", size: 24 },
];

// Border radius data
const borderRadiusTokens = [
  { token: "none", value: "0px", tailwind: "rounded-none", usage: "Sharp corners" },
  { token: "sm", value: "4px", tailwind: "rounded-sm", usage: "Small elements" },
  { token: "md", value: "6px", tailwind: "rounded-md", usage: "Inputs, buttons" },
  { token: "lg", value: "8px", tailwind: "rounded-lg", usage: "Base radius" },
  { token: "xl", value: "12px", tailwind: "rounded-xl", usage: "Cards, containers" },
  { token: "2xl", value: "16px", tailwind: "rounded-2xl", usage: "Large containers" },
  { token: "3xl", value: "20px", tailwind: "rounded-3xl", usage: "Feature cards" },
  { token: "full", value: "9999px", tailwind: "rounded-full", usage: "Pills, avatars, badges" },
];

// Shadow data
const shadowTokens = [
  { token: "xs", tailwind: "shadow-xs", usage: "Inputs, buttons" },
  { token: "sm", tailwind: "shadow-sm", usage: "Cards" },
  { token: "md", tailwind: "shadow-md", usage: "Dropdowns, popovers" },
  { token: "lg", tailwind: "shadow-lg", usage: "Modals, dialogs" },
  { token: "xl", tailwind: "shadow-xl", usage: "Overlays" },
  { token: "2xl", tailwind: "shadow-2xl", usage: "Elevated elements" },
];

// Spacing data
const spacingTokens = [
  { token: "0", value: "0px", tailwind: "p-0, m-0, gap-0" },
  { token: "1", value: "4px", tailwind: "p-1, m-1, gap-1" },
  { token: "2", value: "8px", tailwind: "p-2, m-2, gap-2" },
  { token: "3", value: "12px", tailwind: "p-3, m-3, gap-3" },
  { token: "4", value: "16px", tailwind: "p-4, m-4, gap-4" },
  { token: "5", value: "20px", tailwind: "p-5, m-5, gap-5" },
  { token: "6", value: "24px", tailwind: "p-6, m-6, gap-6" },
  { token: "8", value: "32px", tailwind: "p-8, m-8, gap-8" },
  { token: "10", value: "40px", tailwind: "p-10, m-10, gap-10" },
  { token: "12", value: "48px", tailwind: "p-12, m-12, gap-12" },
  { token: "16", value: "64px", tailwind: "p-16, m-16, gap-16" },
  { token: "20", value: "80px", tailwind: "p-20, m-20, gap-20" },
  { token: "24", value: "96px", tailwind: "p-24, m-24, gap-24" },
];

export default function DesignSystemPage() {
  const { brand, brandName } = useBrand();

  const fontConfig = useMemo(() => brandFontConfigs[brand], [brand]);

  const colorPalettes = useMemo(() => ({
    ...brandColorPalettes[brand],
    ...sharedColorPalettes,
  }), [brand]);

  const brandTokens = useMemo(() => brandTokenConfigs[brand], [brand]);

  const handleDownloadCSS = useCallback(() => {
    const css = generateCSS(brandTokens);
    downloadFile(css, `${brand}-design-tokens.css`, "text/css");
  }, [brand, brandTokens]);

  const handleDownloadJSON = useCallback(() => {
    const json = generateJSON(brandTokens);
    downloadFile(json, `${brand}-design-tokens.json`, "application/json");
  }, [brand, brandTokens]);

  return (
    <TwoColumnLayout
      title="Design System"
      sidebarSections={designSystemSidebarSections}
    >
      {/* Download Design Tokens */}
      <section id="download-tokens" className="mb-12">
        <h2 className="mb-6 text-2xl font-semibold">Download Design Tokens</h2>
        <p className="mb-6 text-muted-foreground">
          Export {brandName} design tokens for use in Figma or your development environment.
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
                  JSON tokens for {brandName} compatible with Tokens Studio plugin
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    All color palettes (Primary, Secondary, Gray, Status)
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
                    {brandName} brand theme
                  </li>
                </ul>
                <Button className="w-full" onClick={handleDownloadJSON}>
                  <Download className="mr-2 h-4 w-4" />
                  Download JSON Tokens
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
                  CSS variables for {brandName} ready for any project
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
                    Light and dark mode semantics
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    {brandName} brand theme
                  </li>
                </ul>
                <Button variant="outline" className="w-full" onClick={handleDownloadCSS}>
                  <Download className="mr-2 h-4 w-4" />
                  Download CSS Variables
                </Button>
                <p className="text-xs text-muted-foreground">
                  Import with:{" "}
                  <code className="rounded bg-muted px-1 py-0.5">
                    @import &apos;{brand}-design-tokens.css&apos;;
                  </code>
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </section>

      {/* Semantic Color Reference */}
      <section id="semantic-colors" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Semantic Color Reference</h2>
        <p className="mb-8 text-muted-foreground">
          Design tokens that map to different values in light and dark mode.
          Use these semantic classes instead of hardcoded colors to ensure
          proper theme support.
        </p>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={defaultViewport}
          variants={staggerContainer}
          className="space-y-10"
        >
          {semanticColorGroups.map((group) => (
            <motion.div key={group.title} variants={fadeIn}>
              <h3 className="mb-4 text-lg font-semibold">{group.title}</h3>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {group.colors.map((color) => (
                  <div
                    key={color.variable}
                    className="rounded-xl border border-border bg-card p-4"
                  >
                    {/* Swatch row: light + dark side by side */}
                    <div className="mb-3 flex gap-2">
                      <div className="flex-1 rounded-lg border border-border overflow-hidden">
                        <div
                          className="h-10 w-full"
                          style={{ backgroundColor: color.light }}
                        />
                        <div className="bg-muted px-2 py-0.5 text-center">
                          <span className="font-mono text-[10px] text-muted-foreground">
                            {color.light}
                          </span>
                        </div>
                      </div>
                      <div className="flex-1 rounded-lg border border-border overflow-hidden">
                        <div
                          className="h-10 w-full"
                          style={{ backgroundColor: color.dark }}
                        />
                        <div className="bg-muted px-2 py-0.5 text-center">
                          <span className="font-mono text-[10px] text-muted-foreground">
                            {color.dark}
                          </span>
                        </div>
                      </div>
                    </div>
                    {/* Labels */}
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-2">
                      <span>Light</span>
                      <span>Dark</span>
                    </div>
                    <p className="text-sm font-medium">{color.name}</p>
                    <code className="text-xs text-muted-foreground">
                      {color.tailwind}
                    </code>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Color Palettes */}
      <section id="color-palettes" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Color Palettes</h2>
        <p className="mb-8 text-muted-foreground">
          Colors extracted from your Figma design system. Each palette includes
          shades from 50 (lightest) to 950 (darkest).
        </p>

        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="space-y-8"
        >
          {Object.entries(colorPalettes).map(([paletteName, colors], index) => (
            <motion.div
              key={paletteName}
              variants={fadeIn}
              transition={{ delay: index * 0.1 }}
            >
              <h3 className="mb-1 text-xl font-semibold">{paletteName}</h3>
              {paletteName === "Secondary" && (
                <p className="mb-3 text-sm text-muted-foreground">
                  Used as accent colour in light mode. Becomes the primary colour in dark mode.
                </p>
              )}
              <div className={`grid grid-cols-11 gap-2 ${paletteName !== "Secondary" ? "mt-3" : ""}`}>
                {colors.map((color) => (
                  <motion.div
                    key={color.name}
                    whileHover={{ scale: 1.05 }}
                    className="text-center"
                  >
                    <div
                      className="mb-2 h-16 w-full rounded-lg border border-border"
                      style={{ backgroundColor: color.hex }}
                    />
                    <span className="block text-xs text-muted-foreground">
                      {color.name}
                    </span>
                    <span className="block font-mono text-[10px] text-muted-foreground">
                      {color.hex}
                    </span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Typography Scale */}
      <section id="typography" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Typography Scale</h2>

        {/* Heading Sub-section */}
        <div className="mb-10">
          <h3 className="mb-2 text-lg font-semibold">Heading</h3>
          <p className="mb-6 text-muted-foreground">
            <Link
              href={fontConfig.displayFontUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              {fontConfig.displayFont} font family
            </Link>{" "}
            is used for home page section titles, page headers, and important text
            elements to create a clear visual hierarchy and improve user navigation.
            Not applicable for rich-text.
          </p>

          {/* Font Preview Card */}
          <motion.div variants={fadeIn}>
            <Card className="border-border bg-background">
              <CardContent className="p-8">
                <span className="mb-4 block text-sm text-muted-foreground">Font family</span>
                <p
                  className="mb-6 text-6xl font-semibold text-foreground md:text-7xl"
                  style={{ fontFamily: fontConfig.displayFontFamily }}
                >
                  {fontConfig.displayFont}
                </p>
                <div
                  className="space-y-3 text-2xl font-light text-muted-foreground md:text-[28px]"
                  style={{ fontFamily: fontConfig.displayFontFamily }}
                >
                  <p>ABCDEFGHIJKLMNOPQRSTUVWXYZ</p>
                  <p>The quick brown fox jumps over the lazy dog.</p>
                  <p>1234567890~!@#$%^&*()-+</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Font Size and Weight Sub-section */}
        <div className="mb-10">
          <h3 className="mb-2 text-lg font-semibold">Font size and weight</h3>
          <p className="mb-6 text-muted-foreground">
            Each sizes includes font weight in Regular (400), Medium (500), and Semibold (600).
          </p>

          <motion.div variants={fadeIn}>
            <Card>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-4 py-3 text-left font-semibold">Name</th>
                        <th className="px-4 py-3 text-left font-semibold">HTML Tag</th>
                        <th className="px-4 py-3 text-left font-semibold">Font Size</th>
                        <th className="px-4 py-3 text-left font-semibold">Line Height</th>
                      </tr>
                    </thead>
                    <tbody>
                      {headingTypography.map((item, index) => (
                        <motion.tr
                          key={item.name}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className="border-b border-border last:border-b-0"
                        >
                          <td className="px-4 py-3 font-semibold">{item.name}</td>
                          <td className="px-4 py-3">
                            {item.tag ? (
                              <code className="rounded bg-muted px-2 py-1 font-mono text-xs">
                                &lt;{item.tag}&gt;
                              </code>
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                          <td className="px-4 py-3 font-mono text-muted-foreground">
                            {item.fontSize}
                          </td>
                          <td className="px-4 py-3 font-mono text-muted-foreground">
                            {item.lineHeight}
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Body Sub-section */}
        <div className="mb-10">
          <h3 className="mb-2 text-lg font-semibold">Body</h3>
          <p className="mb-6 text-muted-foreground">
            <Link
              href={fontConfig.bodyFontUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              {fontConfig.bodyFont} font family
            </Link>{" "}
            is used for body text, paragraphs, labels, captions, and all general-purpose
            content. Optimised for screen readability at small sizes. Used for rich-text content.
          </p>

          {/* Font Preview Card */}
          <motion.div variants={fadeIn}>
            <Card className="border-border bg-background">
              <CardContent className="p-8">
                <span className="mb-4 block text-sm text-muted-foreground">Font family</span>
                <p
                  className="mb-6 text-6xl font-semibold text-foreground md:text-7xl"
                  style={{ fontFamily: fontConfig.bodyFontFamily }}
                >
                  {fontConfig.bodyFont}
                </p>
                <div
                  className="space-y-3 text-2xl font-light text-muted-foreground md:text-[28px]"
                  style={{ fontFamily: fontConfig.bodyFontFamily }}
                >
                  <p>ABCDEFGHIJKLMNOPQRSTUVWXYZ</p>
                  <p>The quick brown fox jumps over the lazy dog.</p>
                  <p>1234567890~!@#$%^&*()-+</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Body Font Size and Weight Sub-section */}
        <div>
          <h3 className="mb-2 text-lg font-semibold">Font size and weight</h3>
          <p className="mb-6 text-muted-foreground">
            Each sizes includes font weight in Regular (400), Medium (500), and Semibold (600).
          </p>

          <motion.div variants={fadeIn}>
            <Card>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-4 py-3 text-left font-semibold">Name</th>
                        <th className="px-4 py-3 text-left font-semibold">HTML Tag</th>
                        <th className="px-4 py-3 text-left font-semibold">Font Size</th>
                        <th className="px-4 py-3 text-left font-semibold">Line Height</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bodyTypography.map((item, index) => (
                        <motion.tr
                          key={item.name}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className="border-b border-border last:border-b-0"
                        >
                          <td className="px-4 py-3 font-semibold">{item.name}</td>
                          <td className="px-4 py-3">
                            <code className="rounded bg-muted px-2 py-1 font-mono text-xs">
                              &lt;{item.tag}&gt;
                            </code>
                          </td>
                          <td className="px-4 py-3 font-mono text-muted-foreground">
                            {item.fontSize}
                          </td>
                          <td className="px-4 py-3 font-mono text-muted-foreground">
                            {item.lineHeight}
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Icons */}
      <section id="icons" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Icons</h2>
        <p className="mb-8 text-muted-foreground">
          The design system uses Lucide React for iconography. Icons are available
          in multiple sizes and inherit the current text colour.
        </p>

        {/* Icon Sizes Reference */}
        <div className="mb-8">
          <h3 className="mb-4 text-lg font-semibold">Size Reference</h3>
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-wrap items-end gap-8">
                {iconSizes.map((size) => (
                  <div key={size.label} className="flex flex-col items-center gap-2">
                    <Home style={{ width: size.size, height: size.size }} className="text-primary" />
                    <div className="text-center">
                      <div className="text-xs font-medium">{size.label}</div>
                      <div className="text-[10px] text-muted-foreground">{size.size}px</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Icon Grid */}
        <Card>
          <CardHeader>
            <CardTitle>Common Icons</CardTitle>
            <CardDescription>
              Commonly used icons from Lucide React
            </CardDescription>
          </CardHeader>
          <CardContent>
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="grid grid-cols-5 gap-4 md:grid-cols-8 lg:grid-cols-10"
            >
              {iconList.map((item, index) => (
                <motion.div
                  key={item.name}
                  variants={fadeIn}
                  transition={{ delay: index * 0.02 }}
                  className="flex flex-col items-center gap-2 rounded-lg p-3 transition-colors hover:bg-muted"
                >
                  <item.icon className="h-5 w-5 text-foreground" />
                  <span className="text-center text-xs text-muted-foreground">
                    {item.name}
                  </span>
                </motion.div>
              ))}
            </motion.div>
          </CardContent>
        </Card>
      </section>

      {/* Border Radius */}
      <section id="border-radius" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Border Radius</h2>
        <p className="mb-8 text-muted-foreground">
          Consistent border radius tokens used across all components. Based on a
          base radius of 0.5rem (8px).
        </p>

        <Card>
          <CardContent className="pt-6">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="grid grid-cols-2 gap-6 md:grid-cols-4"
            >
              {borderRadiusTokens.map((item, index) => (
                <motion.div
                  key={item.token}
                  variants={fadeIn}
                  transition={{ delay: index * 0.05 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div
                    className={`h-20 w-20 bg-primary ${item.tailwind}`}
                  />
                  <div className="text-center">
                    <div className="text-sm font-medium">{item.token}</div>
                    <div className="font-mono text-xs text-muted-foreground">
                      {item.value}
                    </div>
                    <div className="text-[10px] text-muted-foreground">
                      {item.tailwind}
                    </div>
                    <div className="mt-1 text-[10px] text-muted-foreground">
                      {item.usage}
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </CardContent>
        </Card>
      </section>

      {/* Shadows */}
      <section id="shadows" className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Shadows</h2>
        <p className="mb-8 text-muted-foreground">
          Elevation levels using box shadows to create visual hierarchy and depth.
        </p>

        <Card>
          <CardContent className="pt-6">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="grid grid-cols-2 gap-6 md:grid-cols-3 lg:grid-cols-6"
            >
              {shadowTokens.map((item, index) => (
                <motion.div
                  key={item.token}
                  variants={fadeIn}
                  transition={{ delay: index * 0.05 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div
                    className={`flex h-20 w-full items-center justify-center rounded-lg bg-card ${item.tailwind}`}
                  >
                    <span className="text-xs font-medium text-muted-foreground">
                      {item.token}
                    </span>
                  </div>
                  <div className="text-center">
                    <div className="font-mono text-xs text-muted-foreground">
                      {item.tailwind}
                    </div>
                    <div className="text-[10px] text-muted-foreground">
                      {item.usage}
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </CardContent>
        </Card>
      </section>

      {/* Spacing */}
      <section id="spacing" className="mb-0">
        <h2 className="mb-4 text-2xl font-semibold">Spacing</h2>
        <p className="mb-8 text-muted-foreground">
          A consistent spacing scale based on a 4px grid. Used for padding, margin,
          and gap values throughout the design system.
        </p>

        <Card>
          <CardContent className="pt-6">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={defaultViewport}
              variants={staggerContainer}
              className="space-y-3"
            >
              {spacingTokens.map((item, index) => {
                // Scale the width for visibility (multiply by 3, max 100%)
                const barWidth = Math.min(parseInt(item.value) * 3, 384);
                return (
                  <motion.div
                    key={item.token}
                    variants={fadeIn}
                    transition={{ delay: index * 0.03 }}
                    className="flex items-center gap-4"
                  >
                    <div className="w-12 text-right">
                      <span className="text-sm font-medium">{item.token}</span>
                    </div>
                    <div
                      className="h-6 rounded bg-primary"
                      style={{ width: `${barWidth}px`, minWidth: '4px' }}
                    />
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs text-muted-foreground">
                        {item.value}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {item.tailwind}
                      </span>
                    </div>
                  </motion.div>
                );
              })}
            </motion.div>
          </CardContent>
        </Card>
      </section>
    </TwoColumnLayout>
  );
}
