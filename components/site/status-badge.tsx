import { cn } from "@/lib/utils"

type Tone = "gold" | "blue" | "muted"

const toneStyles: Record<Tone, string> = {
  gold: "border-primary/40 bg-primary/10 text-primary",
  blue: "border-secondary/50 bg-secondary/25 text-secondary-foreground",
  muted: "border-border bg-muted text-muted-foreground",
}

export function StatusBadge({
  children,
  tone = "muted",
  className,
}: {
  children: React.ReactNode
  tone?: Tone
  className?: string
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium uppercase tracking-wider",
        toneStyles[tone],
        className,
      )}
    >
      <span aria-hidden className="size-1.5 rounded-full bg-current" />
      {children}
    </span>
  )
}
