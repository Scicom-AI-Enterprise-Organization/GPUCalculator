import type { Metadata } from "next";
import { Inter, Geist_Mono, Roboto, Open_Sans, Montserrat } from "next/font/google";
import "./globals.css";
import { BrandProvider } from "@/components/brand-provider";

// Scicom fonts
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// EMGS fonts
const roboto = Roboto({
  variable: "--font-roboto",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  display: "swap",
});

const montserrat = Montserrat({
  variable: "--font-montserrat",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "900"],
  display: "swap",
});

// Telekom Malaysia fonts
const openSans = Open_Sans({
  variable: "--font-open-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "GPU Calculator - Scicom AI",
  description: "GPU performance benchmark dashboard and VRAM estimator for LLM inference",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link
          href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600,700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className={`${inter.variable} ${geistMono.variable} ${roboto.variable} ${montserrat.variable} ${openSans.variable} font-sans antialiased`}
      >
        <BrandProvider>{children}</BrandProvider>
      </body>
    </html>
  );
}
