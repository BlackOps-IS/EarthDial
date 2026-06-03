import Link from "next/link"
import { ArrowRight, ShieldCheck, Atom, Cpu, Radar, type LucideIcon } from "lucide-react"
import { type TechArea } from "@/lib/content"
import { Card } from "@/components/ui/card"

const iconBySlug: Record<string, LucideIcon> = {
  "secure-ai": ShieldCheck,
  "post-quantum": Atom,
  "privacy-first-systems": Cpu,
  "public-safety-resilience": Radar,
}

export function TechAreaCard({ area }: { area: TechArea }) {
  const Icon = iconBySlug[area.slug] ?? ShieldCheck
  return (
    <Card className="group flex h-full flex-col p-7 transition-colors hover:border-primary/40">
      <span className="inline-flex size-11 items-center justify-center rounded-lg bg-primary/10 text-primary">
        <Icon className="size-5" aria-hidden />
      </span>
      <p className="mt-5 text-xs font-semibold uppercase tracking-[0.16em] text-primary">
        {area.tagline}
      </p>
      <h3 className="mt-2 font-serif text-xl font-medium leading-snug tracking-tight text-balance">
        {area.name}
      </h3>
      <p className="mt-3 flex-1 text-sm leading-relaxed text-muted-foreground">{area.summary}</p>
      <Link
        href={area.href}
        className="mt-6 inline-flex items-center gap-1.5 text-sm font-medium text-foreground transition-colors hover:text-primary"
      >
        {area.cta}
        <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" aria-hidden />
      </Link>
    </Card>
  )
}
