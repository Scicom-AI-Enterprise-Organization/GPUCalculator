"use client";

import * as React from "react";
import { Check, ChevronDown } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
export type BrandTheme = "scicom" | "emgs" | "telekom";

interface Brand {
  id: BrandTheme;
  name: string;
  description: string;
  primaryColor: string;
  accentColor: string;
  logo: string;
}

export const brands: Brand[] = [
  {
    id: "scicom",
    name: "Scicom",
    description: "Default theme",
    primaryColor: "#26346d",
    accentColor: "#c47a3d",
    logo: "/images/scicom-logo.png",
  },
  {
    id: "emgs",
    name: "EMGS",
    description: "Education Malaysia",
    primaryColor: "#be2946",
    accentColor: "#f6a020",
    logo: "/logos/EMGS Logo Black.png",
  },
  {
    id: "telekom",
    name: "Telekom Malaysia",
    description: "TM Theme",
    primaryColor: "#0066b3",
    accentColor: "#00a0e4",
    logo: "/logos/Telekom_Malaysia_logo.svg",
  },
];

interface BrandSwitcherProps {
  currentBrand: BrandTheme;
  onBrandChange: (brand: BrandTheme) => void;
}

export function BrandSwitcher({ currentBrand, onBrandChange }: BrandSwitcherProps) {
  const selectedBrand = brands.find((b) => b.id === currentBrand) || brands[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          className="rounded-full gap-2 px-4 w-[160px] h-10 bg-card hover:bg-muted border-border"
        >
          <span className="relative flex h-2.5 w-2.5 shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
          </span>
          <div className="flex-1 flex items-center justify-center overflow-hidden">
            <Image
              src={selectedBrand.logo}
              alt={selectedBrand.name}
              width={80}
              height={24}
              className="h-5 w-auto max-w-[80px] object-contain"
            />
          </div>
          <ChevronDown className="h-4 w-4 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        {brands.map((brand) => (
          <DropdownMenuItem
            key={brand.id}
            onClick={() => onBrandChange(brand.id)}
            className="flex items-center justify-between cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <div
                className="h-4 w-4 rounded-full border-2"
                style={{
                  backgroundColor: brand.primaryColor,
                  borderColor: brand.accentColor,
                }}
              />
              <div>
                <div className="font-medium">{brand.name}</div>
                <div className="text-xs text-muted-foreground">
                  {brand.description}
                </div>
              </div>
            </div>
            {currentBrand === brand.id && (
              <Check className="h-4 w-4 text-primary" />
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
