"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { TwoColumnLayout } from "@/components/two-column-layout";
import {
  ArrowLeft,
  LogIn,
  Lock,
  Mail,
  Eye,
  EyeOff,
  UserPlus,
  Globe,
  Smartphone,
  Github,
  Chrome,
  Apple,
  KeyRound,
  ArrowRight,
  Check,
  Building2,
  ShieldCheck,
} from "lucide-react";

const loginSidebarSections = [
  {
    items: [
      { label: "Split Hero Login", href: "#split-hero", icon: LogIn },
      { label: "Centered Card", href: "#centered-card", icon: Lock },
      { label: "Social Only", href: "#social-only", icon: Globe },
      { label: "Create Account", href: "#create-account", icon: UserPlus },
      { label: "Two-Factor Auth", href: "#two-factor", icon: Smartphone },
      { label: "Enterprise SSO", href: "#enterprise-sso", icon: Building2 },
    ],
  },
];

/* ================================================================
   TEMPLATE 1 — Split Hero Login
   ================================================================ */
function SplitHeroLogin() {
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [rememberMe, setRememberMe] = useState(true);
  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="overflow-hidden rounded-xl border border-border shadow-2xl">
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          app.scicom.com/login
        </div>
      </div>

      <div className="flex h-[960px]">
        {/* Left — Hero Image */}
        <div className="relative w-1/2 overflow-hidden">
          <Image src="/images/hero-bg-light.jpg" alt="" fill className="object-cover dark:hidden" />
          <Image src="/images/hero-bg.jpg" alt="" fill className="hidden object-cover dark:block" />
          <div className="absolute inset-0 bg-black/30" />
          <div className="relative z-10 flex h-full flex-col justify-between p-8">
            <div className="text-xl font-bold text-white">scicom</div>
            <div>
              <h3 className="text-2xl font-bold text-white">Welcome back</h3>
              <p className="mt-2 text-sm text-white/80">
                Sign in to access your dashboard, manage projects, and collaborate with your team.
              </p>
              <div className="mt-6 flex gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">2.4k+</div>
                  <div className="text-xs text-white/60">Active users</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">99.9%</div>
                  <div className="text-xs text-white/60">Uptime</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">24/7</div>
                  <div className="text-xs text-white/60">Support</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right — Form */}
        <div className="flex w-1/2 flex-col items-center justify-center bg-background px-10">
          <div className="w-full max-w-[300px]">
            <h3 className="mb-1 text-xl font-semibold text-foreground">Sign in</h3>
            <p className="mb-6 text-sm text-muted-foreground">Enter your credentials to continue</p>

            <div className="mb-3">
              <label className="mb-1.5 block text-xs font-medium text-foreground">Email</label>
              <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("email")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "email" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                <Mail className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">name@company.com</span>
              </motion.div>
            </div>

            <div className="mb-3">
              <div className="mb-1.5 flex items-center justify-between">
                <label className="text-xs font-medium text-foreground">Password</label>
                <span className="cursor-pointer text-xs text-primary hover:underline">Forgot?</span>
              </div>
              <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("password")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "password" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                <Lock className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">••••••••</span>
                <div className="ml-auto cursor-pointer" onClick={(e) => { e.stopPropagation(); setShowPassword(!showPassword); }}>
                  {showPassword ? <EyeOff className="h-4 w-4 text-muted-foreground" /> : <Eye className="h-4 w-4 text-muted-foreground" />}
                </div>
              </motion.div>
            </div>

            <label className="mb-5 flex cursor-pointer items-center gap-2 text-sm" onClick={() => setRememberMe(!rememberMe)}>
              <div className={`flex h-4 w-4 items-center justify-center rounded border text-[10px] transition-colors ${rememberMe ? "border-primary bg-primary text-primary-foreground" : "border-border"}`}>
                {rememberMe && <Check className="h-3 w-3" />}
              </div>
              <span className="text-muted-foreground">Remember me</span>
            </label>

            <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} className="mb-4 w-full rounded-lg bg-primary py-2.5 text-sm font-medium text-primary-foreground">
              Sign in
            </motion.button>

            <div className="mb-4 flex items-center gap-3">
              <div className="h-px flex-1 bg-border" />
              <span className="text-xs text-muted-foreground">or continue with</span>
              <div className="h-px flex-1 bg-border" />
            </div>

            <div className="flex gap-2">
              {[{ icon: Chrome, label: "Google" }, { icon: Github, label: "GitHub" }, { icon: Apple, label: "Apple" }].map((p) => (
                <motion.div key={p.label} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }} className="flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-border py-2 text-xs text-muted-foreground transition-colors hover:bg-muted">
                  <p.icon className="h-3.5 w-3.5" />
                  {p.label}
                </motion.div>
              ))}
            </div>

            <p className="mt-5 text-center text-xs text-muted-foreground">
              Don&apos;t have an account?{" "}
              <span className="cursor-pointer font-medium text-primary hover:underline">Sign up</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   TEMPLATE 2 — Centered Card Login
   ================================================================ */
function CenteredCardLogin() {
  const [focusedField, setFocusedField] = useState<string | null>(null);

  return (
    <div className="overflow-hidden rounded-xl border border-border shadow-2xl">
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          portal.scicom.com/auth/login
        </div>
      </div>

      <div className="flex h-[960px] items-center justify-center bg-muted/30 p-8">
        <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="w-full max-w-[380px] rounded-2xl border border-border bg-card p-8 shadow-lg">
          <div className="mb-6 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
              <Lock className="h-6 w-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold text-foreground">Welcome back</h3>
            <p className="mt-1 text-sm text-muted-foreground">Enter your email to sign in to your account</p>
          </div>

          <div className="mb-4 grid grid-cols-2 gap-2">
            {[{ icon: Chrome, label: "Google" }, { icon: Github, label: "GitHub" }].map((p) => (
              <motion.div key={p.label} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-border py-2.5 text-sm text-foreground transition-colors hover:bg-muted">
                <p.icon className="h-4 w-4" />
                {p.label}
              </motion.div>
            ))}
          </div>

          <div className="mb-4 flex items-center gap-3">
            <div className="h-px flex-1 bg-border" />
            <span className="text-xs uppercase text-muted-foreground">or</span>
            <div className="h-px flex-1 bg-border" />
          </div>

          <div className="mb-3">
            <label className="mb-1.5 block text-sm font-medium text-foreground">Email</label>
            <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("email")} className={`rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "email" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
              <span className="text-muted-foreground">m@example.com</span>
            </motion.div>
          </div>

          <div className="mb-5">
            <label className="mb-1.5 block text-sm font-medium text-foreground">Password</label>
            <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("password")} className={`rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "password" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
              <span className="text-muted-foreground">••••••••</span>
            </motion.div>
          </div>

          <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} className="mb-3 w-full rounded-lg bg-foreground py-2.5 text-sm font-medium text-background">
            Sign in
          </motion.button>

          <p className="text-center text-sm text-muted-foreground">
            Don&apos;t have an account?{" "}
            <span className="cursor-pointer font-medium text-foreground underline hover:no-underline">Sign up</span>
          </p>
        </motion.div>
      </div>
    </div>
  );
}

/* ================================================================
   TEMPLATE 3 — Social-Only Login
   ================================================================ */
function SocialOnlyLogin() {
  return (
    <div className="overflow-hidden rounded-xl border border-border shadow-2xl">
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          connect.scicom.com/signin
        </div>
      </div>

      <div className="flex h-[960px] items-center justify-center bg-background p-8">
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="w-full max-w-[360px] text-center">
          <div className="mx-auto mb-2 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary">
            <Globe className="h-7 w-7 text-primary-foreground" />
          </div>
          <h3 className="mb-1 text-2xl font-bold text-foreground">Get started</h3>
          <p className="mb-8 text-sm text-muted-foreground">Choose your preferred sign-in method</p>

          <div className="space-y-2.5">
            {[
              { icon: Chrome, label: "Continue with Google" },
              { icon: Apple, label: "Continue with Apple" },
              { icon: Github, label: "Continue with GitHub" },
              { icon: Building2, label: "Continue with SSO" },
            ].map((p) => (
              <motion.div key={p.label} whileHover={{ scale: 1.01, x: 2 }} whileTap={{ scale: 0.99 }} className="flex cursor-pointer items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 text-sm font-medium text-foreground transition-colors hover:bg-muted">
                <p.icon className="h-5 w-5" />
                {p.label}
                <ArrowRight className="ml-auto h-4 w-4 text-muted-foreground" />
              </motion.div>
            ))}
          </div>

          <div className="my-5 flex items-center gap-3">
            <div className="h-px flex-1 bg-border" />
            <span className="text-xs text-muted-foreground">or</span>
            <div className="h-px flex-1 bg-border" />
          </div>

          <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }} className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-border px-4 py-3 text-sm font-medium text-foreground transition-colors hover:bg-muted">
            <Mail className="h-5 w-5" />
            Continue with Email
          </motion.div>

          <p className="mt-6 text-xs text-muted-foreground">
            By continuing, you agree to our{" "}
            <span className="cursor-pointer underline">Terms</span> and{" "}
            <span className="cursor-pointer underline">Privacy Policy</span>.
          </p>
        </motion.div>
      </div>
    </div>
  );
}

/* ================================================================
   TEMPLATE 4 — Create Account / Registration
   ================================================================ */
function CreateAccountForm() {
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [step, setStep] = useState(1);
  const [agreedTerms, setAgreedTerms] = useState(false);

  const passwordStrength = 3;

  return (
    <div className="overflow-hidden rounded-xl border border-border shadow-2xl">
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          app.scicom.com/register
        </div>
      </div>

      <div className="flex h-[960px]">
        {/* Left — Branding */}
        <div className="flex w-2/5 flex-col justify-between bg-foreground p-8 text-background">
          <div>
            <div className="mb-8 text-lg font-bold">scicom</div>
            <h3 className="text-2xl font-bold">Start building today</h3>
            <p className="mt-2 text-sm opacity-70">Create your account and get started with our design system in minutes.</p>
          </div>
          <div className="space-y-4">
            {[
              { title: "Free to start", desc: "No credit card required" },
              { title: "Full access", desc: "All components and templates" },
              { title: "Team ready", desc: "Invite unlimited members" },
            ].map((item) => (
              <div key={item.title} className="flex items-start gap-3">
                <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary text-[10px]">
                  <Check className="h-3 w-3 text-primary-foreground" />
                </div>
                <div>
                  <div className="text-sm font-medium">{item.title}</div>
                  <div className="text-xs opacity-60">{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right — Form */}
        <div className="flex w-3/5 flex-col justify-center bg-background px-12">
          <div className="w-full max-w-[360px]">
            <div className="mb-6 flex gap-2">
              {[1, 2, 3].map((s) => (
                <div key={s} className={`h-1.5 flex-1 rounded-full transition-colors ${s <= step ? "bg-primary" : "bg-muted"}`} />
              ))}
            </div>

            <h3 className="mb-1 text-xl font-semibold text-foreground">
              {step === 1 ? "Create your account" : step === 2 ? "Set your password" : "Almost done"}
            </h3>
            <p className="mb-6 text-sm text-muted-foreground">
              {step === 1 ? "Enter your details to get started" : step === 2 ? "Choose a strong password" : "Review and confirm"}
            </p>

            {step === 1 && (
              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                <div className="mb-3 grid grid-cols-2 gap-3">
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-foreground">First name</label>
                    <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("first")} className={`rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "first" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                      <span className="text-muted-foreground">John</span>
                    </motion.div>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-foreground">Last name</label>
                    <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("last")} className={`rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "last" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                      <span className="text-muted-foreground">Doe</span>
                    </motion.div>
                  </div>
                </div>
                <div className="mb-3">
                  <label className="mb-1.5 block text-xs font-medium text-foreground">Email</label>
                  <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("email")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "email" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                    <Mail className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">john@company.com</span>
                  </motion.div>
                </div>
                <div className="mb-5">
                  <label className="mb-1.5 block text-xs font-medium text-foreground">Company</label>
                  <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("company")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "company" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                    <Building2 className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Acme Inc.</span>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {step === 2 && (
              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                <div className="mb-3">
                  <label className="mb-1.5 block text-xs font-medium text-foreground">Password</label>
                  <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("pw")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "pw" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                    <Lock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">••••••••••</span>
                  </motion.div>
                  <div className="mt-2 flex gap-1">
                    {[1, 2, 3, 4].map((s) => (
                      <div key={s} className={`h-1 flex-1 rounded-full ${s <= passwordStrength ? (passwordStrength >= 3 ? "bg-green-500" : "bg-yellow-500") : "bg-muted"}`} />
                    ))}
                  </div>
                  <p className="mt-1 text-xs text-green-500">Strong password</p>
                </div>
                <div className="mb-5">
                  <label className="mb-1.5 block text-xs font-medium text-foreground">Confirm password</label>
                  <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("cpw")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "cpw" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                    <Lock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">••••••••••</span>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {step === 3 && (
              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                <div className="mb-4 space-y-3 rounded-lg border border-border bg-muted/30 p-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Name</span>
                    <span className="font-medium text-foreground">John Doe</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Email</span>
                    <span className="font-medium text-foreground">john@company.com</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Company</span>
                    <span className="font-medium text-foreground">Acme Inc.</span>
                  </div>
                </div>
                <label className="mb-5 flex cursor-pointer items-start gap-2 text-sm" onClick={() => setAgreedTerms(!agreedTerms)}>
                  <div className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[10px] transition-colors ${agreedTerms ? "border-primary bg-primary text-primary-foreground" : "border-border"}`}>
                    {agreedTerms && <Check className="h-3 w-3" />}
                  </div>
                  <span className="text-muted-foreground">
                    I agree to the <span className="cursor-pointer text-primary underline">Terms of Service</span> and <span className="cursor-pointer text-primary underline">Privacy Policy</span>
                  </span>
                </label>
              </motion.div>
            )}

            <div className="flex gap-2">
              {step > 1 && (
                <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} onClick={() => setStep(step - 1)} className="rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted">
                  Back
                </motion.button>
              )}
              <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} onClick={() => step < 3 && setStep(step + 1)} className="flex-1 rounded-lg bg-primary py-2.5 text-sm font-medium text-primary-foreground">
                {step === 3 ? "Create account" : "Continue"}
              </motion.button>
            </div>

            <p className="mt-4 text-center text-xs text-muted-foreground">
              Already have an account?{" "}
              <span className="cursor-pointer font-medium text-primary hover:underline">Sign in</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   TEMPLATE 5 — Two-Factor Authentication
   ================================================================ */
function TwoFactorAuth() {
  const [code, setCode] = useState(["", "", "", "", "", ""]);
  const [method, setMethod] = useState<"app" | "sms" | "email">("app");

  const handleCodeClick = (index: number) => {
    const newCode = [...code];
    newCode[index] = newCode[index] ? "" : String(Math.floor(Math.random() * 10));
    setCode(newCode);
  };

  return (
    <div className="overflow-hidden rounded-xl border border-border shadow-2xl">
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          app.scicom.com/verify
        </div>
      </div>

      <div className="flex h-[960px] items-center justify-center bg-background p-8">
        <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="w-full max-w-[400px] text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <ShieldCheck className="h-8 w-8 text-primary" />
          </div>
          <h3 className="mb-1 text-xl font-semibold text-foreground">Two-factor authentication</h3>
          <p className="mb-6 text-sm text-muted-foreground">Enter the verification code from your authenticator app</p>

          <div className="mx-auto mb-6 inline-flex rounded-lg bg-muted p-1">
            {([
              { key: "app", icon: Smartphone, label: "App" },
              { key: "sms", icon: Smartphone, label: "SMS" },
              { key: "email", icon: Mail, label: "Email" },
            ] as const).map((m) => (
              <button key={m.key} onClick={() => setMethod(m.key)} className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${method === m.key ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
                <m.icon className="h-3 w-3" />
                {m.label}
              </button>
            ))}
          </div>

          <div className="mb-6 flex justify-center gap-2">
            {code.map((digit, i) => (
              <motion.div key={i} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => handleCodeClick(i)} className={`flex h-12 w-10 cursor-pointer items-center justify-center rounded-lg border-2 text-lg font-bold transition-all ${digit ? "border-primary bg-primary/5 text-foreground" : "border-border text-muted-foreground"}`}>
                {digit || "·"}
              </motion.div>
            ))}
          </div>

          <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} className="mb-3 w-full max-w-[280px] rounded-lg bg-primary py-2.5 text-sm font-medium text-primary-foreground">
            Verify
          </motion.button>

          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              Didn&apos;t receive a code?{" "}
              <span className="cursor-pointer font-medium text-primary hover:underline">Resend</span>
            </p>
            <p className="text-xs text-muted-foreground">
              <span className="cursor-pointer font-medium text-primary hover:underline">Use recovery code</span>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

/* ================================================================
   TEMPLATE 6 — Enterprise SSO Login
   ================================================================ */
function EnterpriseSSOLogin() {
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [showEmail, setShowEmail] = useState(false);

  return (
    <div className="overflow-hidden rounded-xl border border-border shadow-2xl">
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500/80" />
          <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <div className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <div className="ml-3 flex-1 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground">
          enterprise.scicom.com/sso
        </div>
      </div>

      <div className="flex h-[960px]">
        {/* Left — Branded Panel */}
        <div className="relative flex w-1/2 items-center justify-center overflow-hidden bg-primary">
          <div className="absolute inset-0 opacity-10">
            {Array.from({ length: 20 }).map((_, i) => (
              <div key={i} className="absolute h-32 w-32 rounded-full border border-white" style={{ top: `${(i * 37) % 100}%`, left: `${(i * 53) % 100}%`, transform: `scale(${0.5 + (i % 4) * 0.5})` }} />
            ))}
          </div>
          <div className="relative z-10 px-10 text-center">
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/20 backdrop-blur">
              <Building2 className="h-8 w-8 text-primary-foreground" />
            </div>
            <h3 className="text-2xl font-bold text-primary-foreground">Enterprise Portal</h3>
            <p className="mt-2 text-sm text-primary-foreground/70">Secure single sign-on for your organization</p>
            <div className="mx-auto mt-8 flex max-w-[200px] items-center gap-3 rounded-lg bg-white/10 px-4 py-3 backdrop-blur">
              <ShieldCheck className="h-5 w-5 text-primary-foreground" />
              <div className="text-left text-xs text-primary-foreground">
                <div className="font-medium">SOC 2 Compliant</div>
                <div className="opacity-70">Enterprise grade security</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right — SSO Form */}
        <div className="flex w-1/2 flex-col items-center justify-center bg-background px-10">
          <div className="w-full max-w-[320px]">
            <h3 className="mb-1 text-xl font-semibold text-foreground">Single Sign-On</h3>
            <p className="mb-6 text-sm text-muted-foreground">Sign in with your corporate identity provider</p>

            {!showEmail ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <div className="space-y-2.5">
                  {[
                    { icon: Building2, label: "Continue with Okta", color: "bg-[#007DC1]/10 text-[#007DC1] border-[#007DC1]/20" },
                    { icon: KeyRound, label: "Continue with Azure AD", color: "bg-[#0078D4]/10 text-[#0078D4] border-[#0078D4]/20" },
                    { icon: ShieldCheck, label: "Continue with SAML", color: "bg-primary/10 text-primary border-primary/20" },
                  ].map((sso) => (
                    <motion.div key={sso.label} whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }} className={`flex cursor-pointer items-center gap-3 rounded-lg border px-4 py-3 text-sm font-medium transition-colors hover:bg-muted ${sso.color}`}>
                      <sso.icon className="h-5 w-5" />
                      {sso.label}
                    </motion.div>
                  ))}
                </div>

                <div className="my-5 flex items-center gap-3">
                  <div className="h-px flex-1 bg-border" />
                  <span className="text-xs text-muted-foreground">or</span>
                  <div className="h-px flex-1 bg-border" />
                </div>

                <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} onClick={() => setShowEmail(true)} className="w-full rounded-lg border border-border py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted">
                  Sign in with email
                </motion.button>
              </motion.div>
            ) : (
              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                <div className="mb-3">
                  <label className="mb-1.5 block text-xs font-medium text-foreground">Work email</label>
                  <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("email")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "email" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                    <Mail className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">you@company.com</span>
                  </motion.div>
                </div>
                <div className="mb-5">
                  <label className="mb-1.5 block text-xs font-medium text-foreground">Password</label>
                  <motion.div whileTap={{ scale: 0.995 }} onClick={() => setFocusedField("password")} className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${focusedField === "password" ? "border-primary ring-2 ring-primary/20" : "border-border"}`}>
                    <Lock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">••••••••</span>
                  </motion.div>
                </div>
                <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }} className="mb-3 w-full rounded-lg bg-primary py-2.5 text-sm font-medium text-primary-foreground">
                  Sign in
                </motion.button>
                <button onClick={() => setShowEmail(false)} className="w-full text-center text-xs text-muted-foreground hover:text-foreground">
                  ← Back to SSO options
                </button>
              </motion.div>
            )}

            <p className="mt-6 text-center text-xs text-muted-foreground">
              Need help?{" "}
              <span className="cursor-pointer font-medium text-primary hover:underline">Contact IT support</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   PAGE COMPONENT
   ================================================================ */
export default function LoginTemplatesPage() {
  return (
    <TwoColumnLayout title="Login" sidebarSections={loginSidebarSections}>
      <div className="mb-8">
        <Link href="/templates" className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
          Back to Templates
        </Link>
      </div>

      <div className="mb-14">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <LogIn className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Login Templates</h2>
            <p className="text-sm text-muted-foreground">Authentication & access control pages</p>
          </div>
        </div>
        <p className="text-lg text-muted-foreground">
          A collection of login and authentication page templates in different styles. Each template is fully interactive — click fields, toggle states, and navigate through the flows.
        </p>
      </div>

      {/* Template 1 — Split Hero Login */}
      <section id="split-hero" className="mb-16">
        <div className="mb-4 flex items-center gap-3">
          <h3 className="text-2xl font-semibold text-foreground">Split Hero Login</h3>
          <Badge>Popular</Badge>
        </div>
        <p className="mb-6 text-muted-foreground">
          Classic split-screen layout with a branded hero image on the left and authentication form on the right. Features social sign-in, password visibility toggle, and remember me option.
        </p>
        <SplitHeroLogin />
      </section>

      {/* Template 2 — Centered Card */}
      <section id="centered-card" className="mb-16">
        <div className="mb-4 flex items-center gap-3">
          <h3 className="text-2xl font-semibold text-foreground">Centered Card</h3>
          <Badge variant="secondary">Minimal</Badge>
        </div>
        <p className="mb-6 text-muted-foreground">
          Clean, focused authentication card centered on the page. Prioritizes social sign-in with email/password as a secondary option. Ideal for SaaS products.
        </p>
        <CenteredCardLogin />
      </section>

      {/* Template 3 — Social Only */}
      <section id="social-only" className="mb-16">
        <div className="mb-4 flex items-center gap-3">
          <h3 className="text-2xl font-semibold text-foreground">Social Sign-In</h3>
          <Badge variant="outline">Zero Friction</Badge>
        </div>
        <p className="mb-6 text-muted-foreground">
          Passwordless, social-first login page. Multiple identity providers with a clean list layout. Minimizes friction for new user onboarding.
        </p>
        <SocialOnlyLogin />
      </section>

      {/* Template 4 — Create Account */}
      <section id="create-account" className="mb-16">
        <div className="mb-4 flex items-center gap-3">
          <h3 className="text-2xl font-semibold text-foreground">Create Account</h3>
          <Badge>Multi-step</Badge>
        </div>
        <p className="mb-6 text-muted-foreground">
          Multi-step registration flow with progress indicator, password strength meter, and review step. Split layout with feature highlights. Click &quot;Continue&quot; to navigate through steps.
        </p>
        <CreateAccountForm />
      </section>

      {/* Template 5 — Two-Factor Auth */}
      <section id="two-factor" className="mb-16">
        <div className="mb-4 flex items-center gap-3">
          <h3 className="text-2xl font-semibold text-foreground">Two-Factor Authentication</h3>
          <Badge variant="secondary">Security</Badge>
        </div>
        <p className="mb-6 text-muted-foreground">
          Verification code input with multiple method tabs (App, SMS, Email). Click code boxes to fill them in. Includes resend and recovery code options.
        </p>
        <TwoFactorAuth />
      </section>

      {/* Template 6 — Enterprise SSO */}
      <section id="enterprise-sso" className="mb-0">
        <div className="mb-4 flex items-center gap-3">
          <h3 className="text-2xl font-semibold text-foreground">Enterprise SSO</h3>
          <Badge>Enterprise</Badge>
        </div>
        <p className="mb-6 text-muted-foreground">
          Corporate single sign-on page with Okta, Azure AD, and SAML options. Features a branded hero panel with compliance badges. Falls back to email login.
        </p>
        <EnterpriseSSOLogin />
      </section>
    </TwoColumnLayout>
  );
}
