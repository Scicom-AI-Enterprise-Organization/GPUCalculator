import type { BrandTheme } from "@/components/brand-switcher";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface BrandTokens {
  name: string;
  primary: Record<string, string>;
  secondary: Record<string, string>;
  dataviz?: {
    teal?: Record<string, string>;
    violet?: Record<string, string>;
  };
  fonts: {
    display: string;
    sans: string;
    mono: string;
    weightDisplay: string;
    weightHeading: string;
    weightBody: string;
  };
}

// ─── Shared tokens (identical across all brands) ─────────────────────────────

const sharedTokens = {
  gray: {
    "50": "#fafafa",
    "100": "#f4f4f5",
    "200": "#e4e4e7",
    "300": "#d4d4d8",
    "400": "#a1a1aa",
    "500": "#6b6b74",
    "600": "#52525b",
    "700": "#3f3f46",
    "800": "#27272a",
    "850": "#1d1d21",
    "900": "#18181b",
    "950": "#09090b",
  },
  danger: {
    "50": "#fef2f2",
    "100": "#fee2e2",
    "200": "#fecaca",
    "300": "#fca5a5",
    "400": "#f87171",
    "500": "#ef4444",
    "600": "#dc2626",
    "700": "#b91c1c",
    "800": "#991b1b",
    "900": "#7f1d1d",
    "950": "#450a0a",
  },
  success: {
    "50": "#f0fdf4",
    "100": "#dcfce7",
    "200": "#bbf7d0",
    "300": "#83daa3",
    "400": "#4ade80",
    "500": "#22c55e",
    "600": "#16a34a",
    "700": "#15803d",
    "800": "#166534",
    "900": "#14532d",
    "950": "#052e16",
  },
  warning: {
    "50": "#fefce8",
    "100": "#fef9c3",
    "200": "#fef08a",
    "300": "#fde047",
    "400": "#facc15",
    "500": "#eab308",
    "600": "#ca8a04",
    "700": "#a16207",
    "800": "#854d0e",
    "900": "#713f12",
    "950": "#422006",
  },
  spacing: {
    "0": "0px",
    "1": "4px",
    "2": "8px",
    "3": "12px",
    "4": "16px",
    "5": "20px",
    "6": "24px",
    "8": "32px",
    "10": "40px",
    "12": "48px",
    "16": "64px",
    "20": "80px",
    "24": "96px",
  },
  borderRadius: {
    none: "0px",
    sm: "4px",
    md: "6px",
    lg: "8px",
    xl: "12px",
    "2xl": "16px",
    "3xl": "20px",
    full: "9999px",
  },
  headings: {
    "extra-large": { fontSize: "3.75rem", lineHeight: "4.5rem" },
    large: { fontSize: "3rem", lineHeight: "3.75rem" },
    medium: { fontSize: "2.25rem", lineHeight: "2.75rem" },
    small: { fontSize: "1.875rem", lineHeight: "2.375rem" },
    "extra-small": { fontSize: "1.5rem", lineHeight: "2rem" },
    "2x-small": { fontSize: "1.25rem", lineHeight: "1.75rem" },
    "3x-small": { fontSize: "1rem", lineHeight: "1.5rem" },
    "4x-small": { fontSize: "0.875rem", lineHeight: "1.25rem" },
  },
  body: {
    "extra-large": { fontSize: "1.25rem", lineHeight: "1.875rem" },
    large: { fontSize: "1.125rem", lineHeight: "1.75rem" },
    medium: { fontSize: "1rem", lineHeight: "1.5rem" },
    small: { fontSize: "0.875rem", lineHeight: "1.25rem" },
    "extra-small": { fontSize: "0.75rem", lineHeight: "1.125rem" },
    caption: { fontSize: "0.75rem", lineHeight: "1rem" },
    overline: { fontSize: "0.6875rem", lineHeight: "1rem" },
  },
};

// ─── Brand-specific token configs ────────────────────────────────────────────

export const brandTokenConfigs: Record<BrandTheme, BrandTokens> = {
  scicom: {
    name: "Scicom",
    primary: {
      "50": "oklch(0.94 0.026 278.77)",
      "100": "oklch(0.872 0.057 277.54)",
      "200": "oklch(0.755 0.113 273.89)",
      "300": "oklch(0.629 0.176 268.57)",
      "400": "oklch(0.51 0.19 263.65)",
      "500": "oklch(0.381 0.142 263.58)",
      "600": "oklch(0.334 0.124 263.42)",
      "700": "oklch(0.282 0.105 263.55)",
      "800": "oklch(0.237 0.09 263.69)",
      "900": "oklch(0.181 0.067 263.06)",
      "950": "oklch(0.152 0.057 263.91)",
    },
    secondary: {
      "50": "oklch(0.975 0.013 35.83)",
      "100": "oklch(0.942 0.029 39.38)",
      "200": "oklch(0.885 0.061 42.7)",
      "300": "oklch(0.836 0.093 43.84)",
      "400": "oklch(0.778 0.136 48.97)",
      "500": "oklch(0.722 0.171 53.91)",
      "600": "oklch(0.607 0.143 54.12)",
      "700": "oklch(0.491 0.117 54.03)",
      "800": "oklch(0.369 0.088 53.76)",
      "900": "oklch(0.254 0.06 54.14)",
      "950": "oklch(0.202 0.049 52.98)",
    },
    fonts: {
      display: "'Satoshi', ui-sans-serif, system-ui, sans-serif",
      sans: "'Inter', ui-sans-serif, system-ui, sans-serif",
      mono: "'Geist Mono', ui-monospace, monospace",
      weightDisplay: "600",
      weightHeading: "600",
      weightBody: "400",
    },
  },
  emgs: {
    name: "EMGS",
    primary: {
      "50": "#fdedee",
      "100": "#fadbde",
      "200": "#f7babf",
      "300": "#f3929d",
      "400": "#f16a7c",
      "500": "#e73457",
      "600": "#be2946",
      "700": "#911d33",
      "800": "#641121",
      "900": "#3f0712",
      "950": "#290309",
    },
    secondary: {
      "50": "#fff5ee",
      "100": "#feeee3",
      "200": "#fed9be",
      "300": "#fec79a",
      "400": "#fdb160",
      "500": "#f6a020",
      "600": "#c17d17",
      "700": "#915d0e",
      "800": "#613c06",
      "900": "#382102",
      "950": "#221201",
    },
    dataviz: {
      teal: {
        "400": "oklch(0.72 0.12 175)",
        "500": "oklch(0.58 0.12 172)",
      },
      violet: {
        "400": "oklch(0.70 0.15 280)",
        "500": "oklch(0.55 0.18 275)",
      },
    },
    fonts: {
      display: "'Montserrat', ui-sans-serif, system-ui, sans-serif",
      sans: "'Montserrat', ui-sans-serif, system-ui, sans-serif",
      mono: "'Geist Mono', ui-monospace, monospace",
      weightDisplay: "900",
      weightHeading: "900",
      weightBody: "700",
    },
  },
  telekom: {
    name: "Telekom Malaysia",
    primary: {
      "50": "#e6f3ff",
      "100": "#cce7ff",
      "200": "#99cfff",
      "300": "#66b7ff",
      "400": "#339fff",
      "500": "#0087ff",
      "600": "#0066b3",
      "700": "#005299",
      "800": "#003d73",
      "900": "#00294d",
      "950": "#001426",
    },
    secondary: {
      "50": "#e6f3ff",
      "100": "#cce7ff",
      "200": "#99cfff",
      "300": "#66b7ff",
      "400": "#339fff",
      "500": "#00a0e4",
      "600": "#0080b6",
      "700": "#006088",
      "800": "#00405b",
      "900": "#00202d",
      "950": "#001017",
    },
    fonts: {
      display: "'Open Sans', ui-sans-serif, system-ui, sans-serif",
      sans: "'Open Sans', ui-sans-serif, system-ui, sans-serif",
      mono: "'Geist Mono', ui-monospace, monospace",
      weightDisplay: "600",
      weightHeading: "600",
      weightBody: "400",
    },
  },
};

// ─── CSS generator ───────────────────────────────────────────────────────────

function paletteToCSS(
  prefix: string,
  values: Record<string, string>,
  indent = "  "
): string {
  return Object.entries(values)
    .map(([k, v]) => `${indent}--${prefix}-${k}: ${v};`)
    .join("\n");
}

function scaleToCSS(
  label: string,
  scale: Record<string, { fontSize: string; lineHeight: string }>,
  indent = "  "
): string {
  return Object.entries(scale)
    .map(
      ([k, v]) =>
        `${indent}--${label}-${k}-font-size: ${v.fontSize};\n${indent}--${label}-${k}-line-height: ${v.lineHeight};`
    )
    .join("\n");
}

export function generateCSS(tokens: BrandTokens): string {
  const g = sharedTokens;

  return `/**
 * ${tokens.name} Design System — CSS Variables
 * Generated from Scicom Design Hub
 *
 * Usage:  @import '${tokens.name.toLowerCase().replace(/\s+/g, "-")}-design-tokens.css';
 */

:root {
  /* ===== Base ===== */
  --radius: 0.5rem;

  /* ===== Primary Palette — ${tokens.name} ===== */
${paletteToCSS("primary", tokens.primary)}

  /* ===== Secondary Palette — ${tokens.name} ===== */
${paletteToCSS("secondary-c", tokens.secondary)}

  /* ===== Gray ===== */
${paletteToCSS("gray", g.gray)}

  /* ===== Danger ===== */
${paletteToCSS("danger", g.danger)}

  /* ===== Success ===== */
${paletteToCSS("success", g.success)}

  /* ===== Warning ===== */
${paletteToCSS("warning", g.warning)}

${tokens.dataviz ? `
  /* ===== Data Visualization — Derived (Chart-Only) ===== */
${tokens.dataviz.teal ? paletteToCSS("chart-teal", tokens.dataviz.teal) : ""}
${tokens.dataviz.violet ? paletteToCSS("chart-violet", tokens.dataviz.violet) : ""}
` : ""}
  /* ===== Semantic Mappings — Light Mode ===== */
  --background: var(--gray-50);
  --foreground: var(--gray-950);
  --card: #ffffff;
  --card-foreground: var(--gray-950);
  --popover: #ffffff;
  --popover-foreground: var(--gray-950);
  --primary: var(--primary-600);
  --primary-foreground: #ffffff;
  --secondary: var(--gray-100);
  --secondary-foreground: var(--gray-900);
  --muted: var(--gray-100);
  --muted-foreground: var(--gray-500);
  --accent: var(--secondary-c-100);
  --accent-foreground: var(--secondary-c-900);
  --destructive: var(--danger-500);
  --destructive-foreground: #ffffff;
  --success: var(--success-500);
  --success-foreground: #ffffff;
  --warning: var(--warning-500);
  --warning-foreground: var(--gray-950);
  --border: var(--gray-200);
  --input: var(--gray-200);
  --ring: var(--primary-500);

  /* ===== Chart Tokens ===== */
${tokens.dataviz ? `  --chart-1: var(--primary-500);
  --chart-2: var(--chart-teal-500);
  --chart-3: var(--secondary-c-500);
  --chart-4: var(--success-500);
  --chart-5: var(--chart-violet-500);` : `  --chart-1: var(--primary-500);
  --chart-2: var(--success-500);
  --chart-3: var(--warning-500);
  --chart-4: var(--danger-500);
  --chart-5: var(--primary-300);`}

  /* ===== Typography ===== */
  --font-display: ${tokens.fonts.display};
  --font-sans: ${tokens.fonts.sans};
  --font-mono: ${tokens.fonts.mono};
  --font-weight-display: ${tokens.fonts.weightDisplay};
  --font-weight-heading: ${tokens.fonts.weightHeading};
  --font-weight-body: ${tokens.fonts.weightBody};

  /* Heading Scale */
${scaleToCSS("heading", g.headings)}

  /* Body Scale */
${scaleToCSS("body", g.body)}

  /* ===== Spacing ===== */
${Object.entries(g.spacing)
    .map(([k, v]) => `  --spacing-${k}: ${v};`)
    .join("\n")}

  /* ===== Border Radius ===== */
${Object.entries(g.borderRadius)
    .map(([k, v]) => `  --radius-${k}: ${v};`)
    .join("\n")}
}

.dark {
  /* ===== Semantic Mappings — Dark Mode ===== */
  --background: var(--gray-950);
  --foreground: var(--gray-50);
  --card: var(--gray-900);
  --card-foreground: var(--gray-50);
  --popover: var(--gray-900);
  --popover-foreground: var(--gray-50);
  --primary: var(--secondary-c-500);
  --primary-foreground: #ffffff;
  --secondary: var(--gray-800);
  --secondary-foreground: var(--gray-100);
  --muted: var(--gray-800);
  --muted-foreground: var(--gray-400);
  --accent: var(--primary-400);
  --accent-foreground: var(--primary-950);
  --destructive: var(--danger-600);
  --destructive-foreground: #ffffff;
  --success: var(--success-500);
  --success-foreground: #ffffff;
  --warning: var(--warning-500);
  --warning-foreground: var(--gray-950);
  --border: var(--gray-800);
  --input: var(--gray-800);
  --ring: var(--secondary-c-400);

  /* ===== Chart Tokens — Dark ===== */
${tokens.dataviz ? `  --chart-1: var(--primary-400);
  --chart-2: var(--chart-teal-400);
  --chart-3: var(--secondary-c-400);
  --chart-4: var(--success-400);
  --chart-5: var(--chart-violet-400);` : `  --chart-1: var(--secondary-c-400);
  --chart-2: var(--success-400);
  --chart-3: var(--warning-400);
  --chart-4: var(--danger-400);
  --chart-5: var(--secondary-c-300);`}
}
`;
}

// ─── JSON generator ──────────────────────────────────────────────────────────

export function generateJSON(tokens: BrandTokens): string {
  const g = sharedTokens;

  const output = {
    $brand: tokens.name,
    colors: {
      primary: tokens.primary,
      secondary: tokens.secondary,
      gray: g.gray,
      danger: g.danger,
      success: g.success,
      warning: g.warning,
    },
    dataviz: tokens.dataviz || null,
    typography: {
      fontDisplay: tokens.fonts.display,
      fontSans: tokens.fonts.sans,
      fontMono: tokens.fonts.mono,
      weights: {
        display: tokens.fonts.weightDisplay,
        heading: tokens.fonts.weightHeading,
        body: tokens.fonts.weightBody,
      },
      headings: g.headings,
      body: g.body,
    },
    spacing: g.spacing,
    borderRadius: g.borderRadius,
    semantic: {
      light: {
        background: "var(--gray-50)",
        foreground: "var(--gray-950)",
        card: "#ffffff",
        cardForeground: "var(--gray-950)",
        popover: "#ffffff",
        popoverForeground: "var(--gray-950)",
        primary: "var(--primary-600)",
        primaryForeground: "#ffffff",
        secondary: "var(--gray-100)",
        secondaryForeground: "var(--gray-900)",
        muted: "var(--gray-100)",
        mutedForeground: "var(--gray-500)",
        accent: "var(--secondary-c-100)",
        accentForeground: "var(--secondary-c-900)",
        destructive: "var(--danger-500)",
        destructiveForeground: "#ffffff",
        border: "var(--gray-200)",
        input: "var(--gray-200)",
        ring: "var(--primary-500)",
        ...(tokens.dataviz ? {
          chart1: "var(--primary-500)",
          chart2: "var(--chart-teal-500)",
          chart3: "var(--secondary-c-500)",
          chart4: "var(--success-500)",
          chart5: "var(--chart-violet-500)",
        } : {
          chart1: "var(--primary-500)",
          chart2: "var(--success-500)",
          chart3: "var(--warning-500)",
          chart4: "var(--danger-500)",
          chart5: "var(--primary-300)",
        }),
      },
      dark: {
        background: "var(--gray-950)",
        foreground: "var(--gray-50)",
        card: "var(--gray-900)",
        cardForeground: "var(--gray-50)",
        popover: "var(--gray-900)",
        popoverForeground: "var(--gray-50)",
        primary: "var(--secondary-c-500)",
        primaryForeground: "#ffffff",
        secondary: "var(--gray-800)",
        secondaryForeground: "var(--gray-100)",
        muted: "var(--gray-800)",
        mutedForeground: "var(--gray-400)",
        accent: "var(--primary-400)",
        accentForeground: "var(--primary-950)",
        destructive: "var(--danger-600)",
        destructiveForeground: "#ffffff",
        border: "var(--gray-800)",
        input: "var(--gray-800)",
        ring: "var(--secondary-c-400)",
        ...(tokens.dataviz ? {
          chart1: "var(--primary-400)",
          chart2: "var(--chart-teal-400)",
          chart3: "var(--secondary-c-400)",
          chart4: "var(--success-400)",
          chart5: "var(--chart-violet-400)",
        } : {
          chart1: "var(--secondary-c-400)",
          chart2: "var(--success-400)",
          chart3: "var(--warning-400)",
          chart4: "var(--danger-400)",
          chart5: "var(--secondary-c-300)",
        }),
      },
    },
  };

  return JSON.stringify(output, null, 2);
}

// ─── Download helper ─────────────────────────────────────────────────────────

export function downloadFile(
  content: string,
  filename: string,
  mimeType: string
) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
