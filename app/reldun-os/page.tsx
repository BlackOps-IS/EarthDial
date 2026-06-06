import Image from "next/image"
import Link from "next/link"
import { Info } from "lucide-react"
import { cn } from "@/lib/utils"
import { reldunOS, siteConfig } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, Eyebrow, SectionHeading } from "@/components/site/primitives"
import { createPageMetadata } from "@/lib/metadata"

export const metadata = createPageMetadata({
  title: "Reldun OS — Control Starts at the Kernel Boundary",
  description:
    "Reldun OS is a privacy-first, security-focused operating system research initiative from Black Diamond Project Corp, built around the principle that control starts at the kernel boundary.",
  path: "/reldun-os",
})

export default function ReldunOsPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border bg-diamond-grid">
        <Container className="py-16 sm:py-20">
          <div className="grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <Eyebrow>Privacy-First Systems</Eyebrow>
              <h1 className="mt-4 font-serif text-4xl font-medium leading-[1.08] tracking-tight text-balance sm:text-5xl">
                {reldunOS.name}
              </h1>
              <p className="mt-3 text-lg font-medium text-primary">{reldunOS.tagline}</p>
              <p className="mt-5 max-w-xl text-lg leading-relaxed text-muted-foreground">
                {reldunOS.heroCopy}
              </p>
              <div className="mt-7 flex flex-wrap items-center gap-3">
                <Link
                  href="/contact"
                  className={cn(buttonVariants({ variant: "primary", size: "md" }))}
                >
                  Partner With Us
                </Link>
                <Link
                  href="/research"
                  className={cn(buttonVariants({ variant: "outline", size: "md" }))}
                >
                  View Research Areas
                </Link>
              </div>
              <p className="mt-6 inline-flex items-center rounded-full border border-border bg-muted/40 px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
                Status: {reldunOS.status}
              </p>
            </div>
            <div className="relative mx-auto w-full max-w-md">
              <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-2xl shadow-black/40">
                <Image
                  src={siteConfig.reldunImage}
                  alt={siteConfig.reldunImageAlt}
                  width={640}
                  height={640}
                  className="h-auto w-full"
                  priority
                />
              </div>
            </div>
          </div>
        </Container>
      </section>

      {/* Positioning */}
      <section className="border-b border-border py-16 sm:py-20">
        <Container>
          <p className="mx-auto max-w-3xl text-center font-serif text-2xl font-medium leading-snug text-balance sm:text-3xl">
            {reldunOS.positioning}
          </p>
        </Container>
      </section>

      {/* Focus areas */}
      <section className="border-b border-border py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Research Focus Areas"
            title="What Reldun OS Explores"
            description="Reldun OS research is organized around a small set of architectural principles for privacy and control."
          />
          <div className="mt-10 grid gap-5 sm:grid-cols-2">
            {reldunOS.focusAreas.map((area) => (
              <div
                key={area.title}
                className="rounded-xl border border-border bg-card p-6 transition-colors hover:border-primary/40"
              >
                <h3 className="font-serif text-xl font-medium tracking-tight">{area.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{area.body}</p>
              </div>
            ))}
          </div>
        </Container>
      </section>

      {/* Why it belongs */}
      <section className="border-b border-border bg-muted/20 py-16 sm:py-20">
        <Container>
          <div className="mx-auto max-w-3xl">
            <SectionHeading eyebrow="Mission Alignment" title={reldunOS.whyTitle} />
            <p className="mt-6 text-lg leading-relaxed text-muted-foreground">{reldunOS.whyBody}</p>
          </div>
        </Container>
      </section>

      {/* Disclosure */}
      <section className="py-16 sm:py-20">
        <Container>
          <div className="mx-auto flex max-w-3xl items-start gap-3 rounded-lg border border-border bg-muted/40 p-5">
            <Info className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
            <p className="text-sm leading-relaxed text-muted-foreground">{reldunOS.disclosure}</p>
          </div>
        </Container>
      </section>
    </>
  )
}
