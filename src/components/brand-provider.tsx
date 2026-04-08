"use client";

import * as React from "react";
import { BrandTheme } from "./brand-switcher";

interface BrandContextType {
  brand: BrandTheme;
  setBrand: (brand: BrandTheme) => void;
  brandName: string;
}

const BrandContext = React.createContext<BrandContextType | undefined>(undefined);

const brandNames: Record<BrandTheme, string> = {
  scicom: "Scicom",
  emgs: "EMGS",
  telekom: "Telekom Malaysia",
};

// All properties that brand themes may set via inline styles.
// When switching to Scicom we remove them so globals.css takes over cleanly.
const BRAND_PROPS = [
  "--primary-50", "--primary-100", "--primary-200", "--primary-300",
  "--primary-400", "--primary-500", "--primary-600", "--primary-700",
  "--primary-800", "--primary-900", "--primary-950",
  "--secondary-c-50", "--secondary-c-100", "--secondary-c-200", "--secondary-c-300",
  "--secondary-c-400", "--secondary-c-500", "--secondary-c-600", "--secondary-c-700",
  "--secondary-c-800", "--secondary-c-900", "--secondary-c-950",
  "--primary", "--primary-foreground", "--secondary", "--secondary-foreground",
  "--accent", "--accent-foreground", "--ring",
  "--background", "--foreground", "--card", "--card-foreground",
  "--popover", "--popover-foreground", "--muted", "--muted-foreground",
  "--destructive", "--destructive-foreground", "--border", "--input",
  "--chart-1", "--chart-2", "--chart-3", "--chart-4", "--chart-5",
  "--chart-teal-400", "--chart-teal-500", "--chart-violet-400", "--chart-violet-500",
  "--sidebar", "--sidebar-foreground", "--sidebar-primary", "--sidebar-primary-foreground",
  "--sidebar-accent", "--sidebar-accent-foreground", "--sidebar-border", "--sidebar-ring",
  "--font-sans", "--font-display", "--font-weight-display", "--font-weight-heading", "--font-weight-body",
];

function clearInlineProps(root: HTMLElement) {
  for (const prop of BRAND_PROPS) {
    root.style.removeProperty(prop);
  }
}

function applyBrandTheme(brand: BrandTheme) {
  const root = document.documentElement;
  const isDark = root.classList.contains("dark");

  // Always clear previous inline overrides first so CSS cascade works
  clearInlineProps(root);

  switch (brand) {
    case "emgs":
      // Primary palette (EMGS Red)
      root.style.setProperty("--primary-50", "#fdedee");
      root.style.setProperty("--primary-100", "#fadbde");
      root.style.setProperty("--primary-200", "#f7babf");
      root.style.setProperty("--primary-300", "#f3929d");
      root.style.setProperty("--primary-400", "#f16a7c");
      root.style.setProperty("--primary-500", "#e73457");
      root.style.setProperty("--primary-600", "#be2946");
      root.style.setProperty("--primary-700", "#911d33");
      root.style.setProperty("--primary-800", "#641121");
      root.style.setProperty("--primary-900", "#3f0712");
      root.style.setProperty("--primary-950", "#290309");

      // Secondary palette (EMGS Orange/Warm)
      root.style.setProperty("--secondary-c-50", "#fff5ee");
      root.style.setProperty("--secondary-c-100", "#feeee3");
      root.style.setProperty("--secondary-c-200", "#fed9be");
      root.style.setProperty("--secondary-c-300", "#fec79a");
      root.style.setProperty("--secondary-c-400", "#fdb160");
      root.style.setProperty("--secondary-c-500", "#f6a020");
      root.style.setProperty("--secondary-c-600", "#c17d17");
      root.style.setProperty("--secondary-c-700", "#915d0e");
      root.style.setProperty("--secondary-c-800", "#613c06");
      root.style.setProperty("--secondary-c-900", "#382102");
      root.style.setProperty("--secondary-c-950", "#221201");

      // Fonts — Montserrat for both display and body
      root.style.setProperty("--font-sans", "var(--font-montserrat), ui-sans-serif, system-ui, sans-serif");
      root.style.setProperty("--font-display", "var(--font-montserrat), ui-sans-serif, system-ui, sans-serif");
      root.style.setProperty("--font-weight-display", "900");
      root.style.setProperty("--font-weight-heading", "900");
      root.style.setProperty("--font-weight-body", "700");

      // Data visualization tokens (chart-only)
      root.style.setProperty("--chart-teal-400", "oklch(0.72 0.12 175)");
      root.style.setProperty("--chart-teal-500", "oklch(0.58 0.12 172)");
      root.style.setProperty("--chart-violet-400", "oklch(0.70 0.15 280)");
      root.style.setProperty("--chart-violet-500", "oklch(0.55 0.18 275)");

      // EMGS semantic mappings
      if (isDark) {
        root.style.setProperty("--primary", "var(--secondary-c-500)");
        root.style.setProperty("--primary-foreground", "#ffffff");
        root.style.setProperty("--accent", "var(--primary-400)");
        root.style.setProperty("--accent-foreground", "var(--primary-950)");
        root.style.setProperty("--ring", "var(--secondary-c-400)");
        root.style.setProperty("--chart-1", "var(--primary-400)");
        root.style.setProperty("--chart-2", "var(--chart-teal-400)");
        root.style.setProperty("--chart-3", "var(--secondary-c-400)");
        root.style.setProperty("--chart-4", "var(--success-400)");
        root.style.setProperty("--chart-5", "var(--chart-violet-400)");
      } else {
        root.style.setProperty("--primary", "var(--primary-600)");
        root.style.setProperty("--primary-foreground", "#ffffff");
        root.style.setProperty("--accent", "var(--secondary-c-100)");
        root.style.setProperty("--accent-foreground", "var(--secondary-c-900)");
        root.style.setProperty("--ring", "var(--primary-500)");
        root.style.setProperty("--chart-1", "var(--primary-500)");
        root.style.setProperty("--chart-2", "var(--chart-teal-500)");
        root.style.setProperty("--chart-3", "var(--secondary-c-500)");
        root.style.setProperty("--chart-4", "var(--success-500)");
        root.style.setProperty("--chart-5", "var(--chart-violet-500)");
      }
      break;

    case "telekom":
      // Telekom Malaysia - TM Blue theme
      root.style.setProperty("--primary-50", "#e6f3ff");
      root.style.setProperty("--primary-100", "#cce7ff");
      root.style.setProperty("--primary-200", "#99cfff");
      root.style.setProperty("--primary-300", "#66b7ff");
      root.style.setProperty("--primary-400", "#339fff");
      root.style.setProperty("--primary-500", "#0087ff");
      root.style.setProperty("--primary-600", "#0066b3");
      root.style.setProperty("--primary-700", "#005299");
      root.style.setProperty("--primary-800", "#003d73");
      root.style.setProperty("--primary-900", "#00294d");
      root.style.setProperty("--primary-950", "#001426");
      root.style.setProperty("--primary", "#0066b3");
      root.style.setProperty("--ring", "#00a0e4");
      root.style.setProperty("--accent", "#e6f3ff");
      root.style.setProperty("--accent-foreground", "#00294d");
      // Telekom Malaysia Fonts - Open Sans
      root.style.setProperty("--font-sans", "var(--font-open-sans), ui-sans-serif, system-ui, sans-serif");
      root.style.setProperty("--font-display", "var(--font-open-sans), ui-sans-serif, system-ui, sans-serif");
      root.style.setProperty("--font-weight-display", "600");
      root.style.setProperty("--font-weight-heading", "600");
      root.style.setProperty("--font-weight-body", "400");
      break;

    case "scicom":
    default:
      // Scicom: all colors are defined in globals.css (:root and .dark)
      // No inline overrides needed — CSS cascade handles light/dark mode
      break;
  }
}

export function BrandProvider({ children }: { children: React.ReactNode }) {
  const [brand, setBrandState] = React.useState<BrandTheme>("scicom");

  const setBrand = React.useCallback((newBrand: BrandTheme) => {
    setBrandState(newBrand);
    applyBrandTheme(newBrand);
  }, []);

  // Watch for dark mode class changes and re-apply brand theme
  React.useEffect(() => {
    const observer = new MutationObserver(() => {
      applyBrandTheme(brand);
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, [brand]);

  const value = React.useMemo(
    () => ({
      brand,
      setBrand,
      brandName: brandNames[brand],
    }),
    [brand, setBrand]
  );

  return (
    <BrandContext.Provider value={value}>{children}</BrandContext.Provider>
  );
}

export function useBrand() {
  const context = React.useContext(BrandContext);
  if (context === undefined) {
    throw new Error("useBrand must be used within a BrandProvider");
  }
  return context;
}
