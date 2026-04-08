"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Bot,
  Send,
  Mic,
  Volume2,
  Sparkles,
  Headphones,
  ListChecks,
  MessageSquare,
  ListTree,
  Terminal,
  MessagesSquare,
} from "lucide-react";
import { TwoColumnLayout } from "@/components/two-column-layout";
import { ComponentCard } from "@/components/component-card";

const aiComponentsSidebarSections = [
  {
    items: [
      { label: "Traditional Chatbot", href: "#traditional-chatbot", icon: MessageSquare },
      { label: "Structured Chatbot", href: "#structured-chatbot", icon: ListTree },
      { label: "Headless Chatbot", href: "#headless-chatbot", icon: Terminal },
      { label: "Agent Assist", href: "#agent-assist", icon: Headphones },
      { label: "Audio Response", href: "#audio-response", icon: Volume2 },
      { label: "AI Message Bubbles", href: "#ai-message-bubbles", icon: MessagesSquare },
    ],
  },
];

export default function AIComponentsPage() {
  return (
    <TwoColumnLayout
      title="AI Components"
      sidebarSections={aiComponentsSidebarSections}
    >
      {/* Introduction */}
      <div className="mb-12">
        <p className="text-lg text-muted-foreground">
          AI-powered components for building intelligent conversational interfaces,
          agent assistance systems, and voice-enabled applications.
        </p>
      </div>

      {/* Traditional Chatbot */}
      <section id="traditional-chatbot" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Traditional Chatbot</h2>
        <p className="mb-6 text-muted-foreground">
          Classic chat interface with message bubbles, input field, and send button.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Basic Chatbot" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-b border-border p-2">
                <Bot className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">AI Assistant</span>
              </div>
              <div className="space-y-2 p-2">
                <div className="flex gap-2">
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <Bot className="h-3 w-3 text-primary" />
                  </div>
                  <div className="rounded-lg bg-muted px-2 py-1 text-xs">
                    How can I help you today?
                  </div>
                </div>
                <div className="flex justify-end">
                  <div className="rounded-lg bg-primary px-2 py-1 text-xs text-white">
                    Show me reports
                  </div>
                </div>
              </div>
              <div className="flex gap-1 border-t border-border p-2">
                <Input placeholder="Type a message..." className="h-6 text-xs" />
                <Button size="icon" className="h-6 w-6">
                  <Send className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="With Typing Indicator" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-b border-border p-2">
                <Bot className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">AI Assistant</span>
                <Badge variant="secondary" className="ml-auto h-4 text-[10px]">Online</Badge>
              </div>
              <div className="space-y-2 p-2">
                <div className="flex gap-2">
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <Bot className="h-3 w-3 text-primary" />
                  </div>
                  <div className="rounded-lg bg-muted px-2 py-1">
                    <div className="flex gap-1">
                      <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground" style={{ animationDelay: "0ms" }} />
                      <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground" style={{ animationDelay: "150ms" }} />
                      <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground" style={{ animationDelay: "300ms" }} />
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex gap-1 border-t border-border p-2">
                <Input placeholder="Type a message..." className="h-6 text-xs" />
                <Button size="icon" className="h-6 w-6">
                  <Send className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Minimized State" count={1}>
            <div className="w-full max-w-[60px] rounded-full bg-primary p-3 shadow-lg cursor-pointer hover:scale-105 transition-transform">
              <Bot className="h-6 w-6 text-white" />
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Structured Chatbot */}
      <section id="structured-chatbot" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Structured Chatbot</h2>
        <p className="mb-6 text-muted-foreground">
          Guided conversation flow with predefined options and leading questions.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="With Quick Actions" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-b border-border p-2">
                <ListChecks className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">Guided Assistant</span>
              </div>
              <div className="space-y-2 p-2">
                <div className="rounded-lg bg-muted px-2 py-1 text-xs">
                  What would you like to do?
                </div>
                <div className="space-y-1">
                  <button className="w-full rounded border border-primary/30 bg-primary/5 px-2 py-1 text-left text-xs text-primary hover:bg-primary/10">
                    Create a new report
                  </button>
                  <button className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted">
                    View analytics
                  </button>
                  <button className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted">
                    Get help
                  </button>
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Multi-Step Flow" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-b border-border p-2">
                <ListChecks className="h-4 w-4 text-primary" />
                <span className="text-xs font-medium">Step 2 of 3</span>
              </div>
              <div className="p-2">
                <div className="mb-2 h-1 w-full rounded-full bg-muted">
                  <div className="h-1 w-2/3 rounded-full bg-primary" />
                </div>
                <div className="rounded-lg bg-muted px-2 py-1 text-xs">
                  Select your preferred format:
                </div>
                <div className="mt-2 space-y-1">
                  <button className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted">
                    PDF Document
                  </button>
                  <button className="w-full rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted">
                    Excel Spreadsheet
                  </button>
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Headless Chatbot */}
      <section id="headless-chatbot" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Headless Chatbot</h2>
        <p className="mb-6 text-muted-foreground">
          AI-generated content that integrates seamlessly into your UI without a chat window.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Inline AI Response" count={1}>
            <div className="w-full max-w-[200px] space-y-2">
              <div className="rounded-lg border border-border bg-card p-2">
                <div className="mb-2 flex items-center gap-1">
                  <Sparkles className="h-3 w-3 text-primary" />
                  <span className="text-xs font-medium">AI Generated</span>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>Revenue</span>
                    <span className="font-medium">$45.2k</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted">
                    <div className="h-1.5 w-[75%] rounded-full bg-primary" />
                  </div>
                </div>
              </div>
              <div className="rounded-lg border border-primary/30 bg-primary/5 p-2">
                <div className="flex items-start gap-2">
                  <Bot className="h-4 w-4 shrink-0 text-primary" />
                  <p className="text-xs text-primary">
                    Based on your request, here&apos;s the revenue breakdown.
                  </p>
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="AI Summary Card" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-3">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-1">
                  <Sparkles className="h-3 w-3 text-primary" />
                  <span className="text-xs font-medium">AI Summary</span>
                </div>
                <Badge variant="secondary" className="h-4 text-[10px]">Auto</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Sales increased by 12% this quarter, driven primarily by new customer acquisitions in the enterprise segment.
              </p>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Agent Assist */}
      <section id="agent-assist" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Agent Assist</h2>
        <p className="mb-6 text-muted-foreground">
          Real-time AI suggestions for customer service agents during live interactions.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Suggested Response" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center justify-between border-b border-border p-2">
                <div className="flex items-center gap-2">
                  <Headphones className="h-4 w-4 text-primary" />
                  <span className="text-xs font-medium">Agent Assist</span>
                </div>
                <Badge variant="secondary" className="h-4 text-[10px]">Live</Badge>
              </div>
              <div className="p-2">
                <div className="mb-2 rounded bg-warning-50 p-1.5 dark:bg-warning-900/30">
                  <div className="flex items-center gap-1 text-[10px] font-medium text-warning-700 dark:text-warning-400">
                    <Sparkles className="h-3 w-3" />
                    Suggested Response
                  </div>
                  <p className="mt-1 text-xs text-warning-600 dark:text-warning-300">
                    I can help reset your password...
                  </p>
                </div>
                <div className="flex gap-1">
                  <Button size="sm" className="h-5 flex-1 text-[10px]">
                    Use
                  </Button>
                  <Button size="sm" variant="outline" className="h-5 flex-1 text-[10px]">
                    Edit
                  </Button>
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Knowledge Suggestion" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card shadow-sm">
              <div className="flex items-center justify-between border-b border-border p-2">
                <div className="flex items-center gap-2">
                  <Headphones className="h-4 w-4 text-primary" />
                  <span className="text-xs font-medium">Knowledge Base</span>
                </div>
              </div>
              <div className="p-2 space-y-2">
                <div className="rounded border border-border p-1.5">
                  <div className="text-[10px] font-medium">Password Reset Policy</div>
                  <p className="text-[10px] text-muted-foreground">Users can reset via email or SMS verification...</p>
                </div>
                <div className="rounded border border-border p-1.5">
                  <div className="text-[10px] font-medium">Account Recovery</div>
                  <p className="text-[10px] text-muted-foreground">For locked accounts, verify identity with...</p>
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* Audio Response */}
      <section id="audio-response" className="mb-12">
        <h2 className="mb-2 text-2xl font-semibold">Audio Response</h2>
        <p className="mb-6 text-muted-foreground">
          Voice playback and recording components for speech-enabled AI interactions.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="Voice Playback" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-3 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4 text-primary" />
                  <span className="text-xs font-medium">Voice Response</span>
                </div>
              </div>
              <div className="mb-3 flex items-center gap-1">
                {[12, 18, 10, 22, 14, 20, 8, 16, 24, 11].map((h, i) => (
                  <div
                    key={i}
                    className="w-1 rounded-full bg-primary/60"
                    style={{ height: `${h}px` }}
                  />
                ))}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">0:12 / 0:45</span>
                <div className="flex gap-1">
                  <Button size="icon" variant="outline" className="h-6 w-6">
                    <Mic className="h-3 w-3" />
                  </Button>
                  <Button size="icon" className="h-6 w-6">
                    <Volume2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="Voice Recording" count={1}>
            <div className="w-full max-w-[200px] rounded-lg border border-border bg-card p-3 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Mic className="h-4 w-4 text-danger-500 animate-pulse" />
                  <span className="text-xs font-medium text-danger-500">Recording...</span>
                </div>
                <span className="text-xs text-muted-foreground">0:05</span>
              </div>
              <div className="mb-3 flex items-center gap-1">
                {[8, 14, 20, 16, 22, 12, 18, 10, 15, 19].map((h, i) => (
                  <div
                    key={i}
                    className="w-1 rounded-full bg-danger-500/60 animate-pulse"
                    style={{ height: `${h}px` }}
                  />
                ))}
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" className="flex-1 h-6 text-[10px]">
                  Cancel
                </Button>
                <Button size="sm" className="flex-1 h-6 text-[10px]">
                  Stop
                </Button>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>

      {/* AI Message Bubbles */}
      <section id="ai-message-bubbles" className="mb-0">
        <h2 className="mb-2 text-2xl font-semibold">AI Message Bubbles</h2>
        <p className="mb-6 text-muted-foreground">
          Styled message components with feedback actions and AI attribution.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          <ComponentCard title="With Feedback" count={1}>
            <div className="w-full max-w-[200px] space-y-2">
              <div className="flex gap-2">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary">
                  <Bot className="h-3 w-3 text-white" />
                </div>
                <div className="space-y-1">
                  <div className="rounded-lg rounded-tl-none bg-muted px-2 py-1 text-xs">
                    I found 3 matching results.
                  </div>
                  <div className="flex gap-1">
                    <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary cursor-pointer hover:bg-primary/20">
                      helpful
                    </span>
                    <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] cursor-pointer hover:bg-muted">
                      not helpful
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex justify-end">
                <div className="rounded-lg rounded-tr-none bg-primary px-2 py-1 text-xs text-white">
                  Thanks! Show me more details
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard title="With Sources" count={1}>
            <div className="w-full max-w-[200px]">
              <div className="flex gap-2">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary">
                  <Bot className="h-3 w-3 text-white" />
                </div>
                <div className="space-y-1">
                  <div className="rounded-lg rounded-tl-none bg-muted px-2 py-1 text-xs">
                    Based on the documentation, the feature was released in v2.0.
                  </div>
                  <div className="text-[10px] text-muted-foreground">
                    Sources: <span className="text-primary cursor-pointer hover:underline">docs.example.com</span>
                  </div>
                </div>
              </div>
            </div>
          </ComponentCard>
        </div>
      </section>
    </TwoColumnLayout>
  );
}
