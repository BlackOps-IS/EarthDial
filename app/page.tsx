import Link from "next/link"
import Image from "next/image"
import { ArrowRight, Radar } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  techAreas,
  trustStrip,
  reldunOS,
  siteConfig,
  leadership,
  faqs,
  missionPrinciples,
} from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, SectionHeading, DiamondMark, Eyebrow } from "@/components/site/primitives"
import { Hero } from "@/components/site/hero"
import { TechAreaCard } from "@/components/site/tech-area-card"
import { FaqSection } from "@/components/site/faq-section"
import { SupportSection } from "@/components/site/support-section"
import { Card } from "@/components/ui/card"

export default function HomePage() {
  return (
    <>
      {/* SECTION 1 — Hero */}
      <Hero />

      {/* SECTION 2 — Trust strip */}
      <section className="border-b border-border bg-[oklch(0.14_0.004_286)]" aria-label="Trust signals">
        <Container>
          <ul className="grid grid-cols-2 divide-x divide-y divide-border border-x border-border sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
            {trustStrip.map((item) => (
              <li
                key={item}
                className="flex min-h-20 min-w-0 items-center gap-3 px-4 py-4 text-xs font-medium leading-snug text-foreground/80 sm:text-sm"
              >
                <DiamondMark className="size-3.5 shrink-0" />
                <span className="min-w-0 text-left">{item}</span>
              </li>
            ))}
          </ul>
        </Container>
      </section>

      {/* SECTION 3 — Mission */}
      <section className="relative overflow-hidden py-20 sm:py-24">
        <Container className="relative grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div>
            <Eyebrow>Our Mission</Eyebrow>
            <h2 className="mt-4 font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
              Technology Built for High-Trust Environments
            </h2>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
              The future of security is not a single product. It is the responsible integration of
              secure intelligence, resilient cryptography, trustworthy computing foundations and
              public-safety awareness. Black Diamond Project Corp brings these research paths
              together under one public-benefit mission.
            </p>
          </div>
          <ul className="border-y border-border">
            {missionPrinciples.map((principle) => (
              <li key={principle.title} className="border-b border-border py-6 last:border-b-0 sm:py-7">
                <div className="flex items-start gap-3.5">
                  <DiamondMark className="mt-0.5 size-5 shrink-0" />
                  <div>
                    <h3 className="font-serif text-lg font-medium tracking-tight">
                      {principle.title}
                    </h3>
                    <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                      {principle.body}
                    </p>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </Container>
      </section>

      {/* SECTION 4 — Research pillars */}
      <section className="border-t border-border bg-[oklch(0.14_0.004_286)] py-20 sm:py-24">
        <Container>
          <SectionHeading
            eyebrow="What We Research"
            title="Four Connected Technology Areas"
            description="One coherent mission: protecting people, protecting information and building resilient systems for the future."
          />
          <div className="mt-12 grid gap-x-8 gap-y-2 sm:grid-cols-2 lg:grid-cols-4">
            {techAreas.map((area) => (
              <TechAreaCard key={area.slug} area={area} />
            ))}
          </div>
        </Container>
      </section>

      {/* SECTION 5 — Featured initiative: Reldun OS */}
      <section className="border-t border-border py-20 sm:py-24">
        <Container className="grid items-center gap-12 lg:grid-cols-2">
          <div className="order-2 flex justify-center overflow-hidden border-y border-border bg-[oklch(0.12_0.004_286)] p-6 sm:p-10 lg:order-1">
            <Image
              src={siteConfig.reldunImage}
              alt={siteConfig.reldunImageAlt}
              width={1024}
              height={1024}
              sizes="(max-width: 1024px) 80vw, 40vw"
              className="h-auto w-full max-w-xs sm:max-w-sm lg:max-w-md"
            />
          </div>
          <div className="order-1 lg:order-2">
            <p className="text-sm font-medium tracking-wide text-primary">
              Featured Initiative
            </p>
            <h2 className="mt-4 font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
              Reldun OS
            </h2>
            <p className="mt-2 text-lg text-foreground/80">{reldunOS.tagline}</p>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-muted-foreground">
              A privacy-first, security-focused operating system research initiative exploring
              secure computing foundations for high-trust environments.
            </p>
            <Link
              href="/reldun-os"
              className={cn(buttonVariants({ variant: "primary", size: "md" }), "mt-7")}
            >
              Explore Reldun OS
              <ArrowRight className="size-4" aria-hidden />
            </Link>
          </div>
        </Container>
      </section>

      {/* SECTION 6 + 7 — EarthDial & Post-Quantum */}
      <section className="border-t border-border bg-[oklch(0.14_0.004_286)] py-20 sm:py-24">
        <Container className="grid items-stretch gap-8 lg:grid-cols-[1.35fr_0.65fr]">
          <div className="relative overflow-hidden border-y border-primary/25 py-10 sm:py-12">
            <div className="relative z-10 max-w-xl">
              <p className="text-sm font-medium tracking-wide text-primary">
                Featured Initiative
              </p>
              <h2 className="mt-4 font-serif text-4xl font-medium tracking-tight text-balance sm:text-5xl">
                EarthDial
              </h2>
              <p className="mt-5 text-lg leading-relaxed text-muted-foreground">
                A public-safety resilience technology initiative focused on emergency awareness,
                preparedness and community-impacting conditions.
              </p>
              <Link
                href="/earthdial"
                className={cn(buttonVariants({ variant: "outline", size: "md" }), "mt-8")}
              >
                Explore EarthDial
                <ArrowRight className="size-4" aria-hidden />
              </Link>
            </div>
            <div
              className="pointer-events-none absolute -right-16 top-1/2 size-64 -translate-y-1/2 opacity-35 sm:right-2 sm:size-72"
              aria-hidden
            >
              <div className="absolute inset-0 rounded-full border border-primary/25" />
              <div className="absolute inset-[18%] rounded-full border border-primary/35" />
              <div className="absolute inset-[36%] rounded-full border border-primary/50" />
              <div className="absolute left-1/2 top-0 h-full w-px bg-primary/25" />
              <div className="absolute left-0 top-1/2 h-px w-full bg-primary/25" />
              <Radar className="absolute left-1/2 top-1/2 size-10 -translate-x-1/2 -translate-y-1/2 text-primary" />
            </div>
          </div>
          <Card className="flex flex-col p-8">
            <p className="text-sm font-medium tracking-wide text-primary">
              Research Focus
            </p>
            <h2 className="mt-4 font-serif text-2xl font-medium tracking-tight text-balance">
              Preparing for Security Beyond Today&apos;s Cryptography
            </h2>
            <p className="mt-4 flex-1 text-base leading-relaxed text-muted-foreground">
              Black Diamond explores responsible approaches to quantum-era security challenges,
              resilient information protection and future-ready trust systems.
            </p>
            <Link
              href="/post-quantum-security"
              className="mt-6 inline-flex items-center gap-1.5 text-sm font-medium text-foreground transition-colors hover:text-primary"
            >
              Explore Post-Quantum Research
              <ArrowRight className="size-4" aria-hidden />
            </Link>
          </Card>
        </Container>
      </section>

      {/* SECTION 8 — Leadership preview */}
      <section className="border-t border-border bg-[oklch(0.14_0.004_286)] py-20 sm:py-24">
        <Container>
          <SectionHeading
            eyebrow="Leadership"
            title="Guided by Research and Responsibility"
            description="Black Diamond Project Corp is led by researchers focused on secure AI, cybersecurity, secure systems and public-benefit technology."
          />
          <div className="mt-12 grid gap-6 sm:grid-cols-2">
            {leadership.map((leader) => (
              <Card key={leader.name} className="p-7">
                <h3 className="font-serif text-xl font-medium tracking-tight">{leader.name}</h3>
                <p className="mt-1 text-sm font-medium text-primary">{leader.role}</p>
                <p className="mt-4 text-sm leading-relaxed text-muted-foreground">{leader.bio}</p>
              </Card>
            ))}
          </div>
          <Link
            href="/mission#leadership"
            className="mt-8 inline-flex items-center gap-1.5 text-sm font-medium text-foreground transition-colors hover:text-primary"
          >
            Read more about our mission and leadership
            <ArrowRight className="size-4" aria-hidden />
          </Link>
        </Container>
      </section>

      {/* SECTION 9 — FAQ */}
      <section className="border-t border-border py-20 sm:py-24">
        <Container className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr]">
          <SectionHeading
            eyebrow="Questions"
            title="Frequently Asked Questions"
            description="Clear answers about the organization, its initiatives, and foundation status."
          />
          <FaqSection items={faqs} />
        </Container>
      </section>

      {/* SECTION 10 — Support */}
      <SupportSection />
    </>
  )
}
