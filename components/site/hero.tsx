import Link from "next/link"
import { ArrowRight, ShieldCheck, Atom, Radar } from "lucide-react"
import { cn } from "@/lib/utils"
import { siteConfig, heroCredibility } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, Eyebrow, DiamondMark } from "./primitives"

const pillars = [
  { icon: ShieldCheck, label: "Trustworthy AI" },
  { icon: Atom, label: "Quantum Resilience" },
  { icon: Radar, label: "Public-Safety Technology" },
]

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-border bg-diamond-grid">
      <Container className="grid items-center gap-12 py-20 lg:grid-cols-12 lg:py-28">
        <div className="lg:col-span-7">
          <Eyebrow>{siteConfig.organizationName}</Eyebrow>
          <h1 className="mt-5 font-serif text-4xl font-medium leading-[1.05] tracking-tight text-balance sm:text-5xl lg:text-6xl">
            Technology Built for{" "}
            <span className="text-gradient-gold">Public Benefit.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
            {siteConfig.organizationName} is a private foundation advancing trustworthy AI,
            quantum-resilient systems, and public-safety technology through responsible research and
            mission-driven innovation.
          </p>

          <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link
              href="/programs"
              className={cn(buttonVariants({ variant: "primary", size: "lg" }))}
            >
              Explore Our Research
              <ArrowRight className="size-4" aria-hidden />
            </Link>
            <Link
              href="/foundation-status"
              className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
            >
              View Foundation Status
            </Link>
          </div>

          <Link
            href="/support"
            className="mt-5 inline-flex items-center gap-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            Support the Mission
            <ArrowRight className="size-3.5" aria-hidden />
          </Link>

          {/* Credibility strip */}
          <ul className="mt-10 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-border pt-6">
            {heroCredibility.map((item) => (
              <li key={item} className="flex items-center gap-2 text-sm text-foreground/80">
                <DiamondMark className="size-3.5" />
                {item}
              </li>
            ))}
          </ul>
        </div>

        {/* Visual panel */}
        <div className="lg:col-span-5">
          <div className="relative rounded-xl border border-border bg-card/60 p-8 backdrop-blur-sm">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent" />
            <DiamondMark className="size-10" />
            <p className="mt-6 font-serif text-xl leading-snug text-balance">
              Research framed around validation, transparency, and reproducible results.
            </p>
            <div className="mt-8 flex flex-col gap-4">
              {pillars.map((pillar) => (
                <div
                  key={pillar.label}
                  className="flex items-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-3"
                >
                  <span className="inline-flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                    <pillar.icon className="size-4" aria-hidden />
                  </span>
                  <span className="text-sm font-medium">{pillar.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Container>
    </section>
  )
}
