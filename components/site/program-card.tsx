import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { type Program, statusStyles } from "@/lib/content"
import { Card, CardContent } from "@/components/ui/card"
import { StatusBadge } from "./status-badge"
import { DiamondMark } from "./primitives"

export function ProgramCard({
  program,
  featured = false,
}: {
  program: Program
  featured?: boolean
}) {
  const status = statusStyles[program.status]

  return (
    <Card
      className={cn(
        "group relative flex flex-col overflow-hidden transition-colors hover:border-primary/40",
        featured && "hairline-top",
      )}
    >
      <CardContent className="flex flex-1 flex-col gap-5 p-7">
        <div className="flex items-start justify-between gap-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            {program.label}
          </p>
          <DiamondMark className="size-5 shrink-0 opacity-70" />
        </div>

        <StatusBadge tone={status.tone}>{status.label}</StatusBadge>

        <h3 className="font-serif text-2xl font-medium leading-tight tracking-tight text-balance">
          {program.name}
        </h3>

        <p className="text-sm leading-relaxed text-muted-foreground">{program.summary}</p>

        {program.footnote ? (
          <p className="mt-auto border-l-2 border-primary/40 pl-3 text-xs leading-relaxed text-muted-foreground/90">
            {program.footnote}
          </p>
        ) : null}

        <Link
          href={program.href}
          className="mt-2 inline-flex items-center gap-1.5 text-sm font-semibold text-primary transition-colors hover:text-primary/80"
        >
          {program.cta}
          <ArrowRight
            className="size-4 transition-transform group-hover:translate-x-0.5"
            aria-hidden
          />
        </Link>
      </CardContent>
    </Card>
  )
}
