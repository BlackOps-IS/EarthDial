import Link from "next/link"
import { ArrowRight, ShieldCheck, Atom, Cpu, Radar, type LucideIcon } from "lucide-react"
import { type TechArea } from "@/lib/content"

const iconBySlug: Record<string, LucideIcon> = {
  "secure-ai": ShieldCheck,
  "post-quantum": Atom,
  "privacy-first-systems": Cpu,
  "public-safety-resilience": Radar,
}

export function TechAreaCard({ area }: { area: TechArea }) {
  const Icon = iconBySlug[area.slug] ?? ShieldCheck
  return (
    <article className="group flex h-full flex-col border-t border-border py-6 transition-colors hover:border-primary">
      <span className="inline-flex size-9 items-center justify-center border border-primary/30 text-primary">
        <Icon className="size-5" aria-hidden />
      </span>
      <p className="mt-5 text-xs font-medium tracking-wide text-primary">
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
    </article>
  )
}
