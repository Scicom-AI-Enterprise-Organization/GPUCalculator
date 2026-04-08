import { AimsTopNavigation } from "@/components/aims/top-navigation";
import { AimsSideNavigation } from "@/components/aims/side-navigation";

export default function AimsAppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Top Navigation */}
      <AimsTopNavigation />

      {/* Body: Side Nav + Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Side Navigation */}
        <AimsSideNavigation />

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
