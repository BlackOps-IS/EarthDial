import Link from "next/link"
import Image from "next/image"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { techAreas, trustStrip, reldunOS, siteConfig, leadership, faqs } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, SectionHeading, DiamondMark } from "@/components/site/primitives"
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
      <section className="border-b border-border bg-[oklch(0.14_0.004_286)]">
        <Container className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3 py-6">
          {trustStrip.map((item) => (
            <span key={item} className="flex items-center gap-2 text-sm font-medium text-foreground/80">
              <DiamondMark className="size-3.5" />
              {item}
            </span>
          ))}
        </Container>
      </section>

      {/* SECTION 3 — Mission */}
      <section className="py-20 sm:py-24">
        <Container>
          <SectionHeading
            eyebrow="Our Mission"
            title="Technology Built for High-Trust Environments"
            description="The future of security is not a single product. It is the responsible integration of secure intelligence, resilient cryptography, trustworthy computing foundations and public-safety awareness. Black Diamond Project Corp brings these research paths together under one public-benefit mission."
          />
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
          <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {techAreas.map((area) => (
              <TechAreaCard key={area.slug} area={area} />
            ))}
          </div>
        </Container>
      </section>

      {/* SECTION 5 — Featured initiative: Reldun OS */}
      <section className="border-t border-border py-20 sm:py-24">
        <Container className="grid items-center gap-12 lg:grid-cols-2">
          <div className="order-2 flex justify-center overflow-hidden rounded-xl border border-border bg-[oklch(0.12_0.004_286)] p-6 sm:p-10 lg:order-1">
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
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
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
        <Container className="grid gap-6 lg:grid-cols-2">
          <Card className="flex flex-col p-8">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
              Featured Initiative
            </p>
            <h2 className="mt-4 font-serif text-2xl font-medium tracking-tight">EarthDial</h2>
            <p className="mt-4 flex-1 text-base leading-relaxed text-muted-foreground">
              A public-safety resilience technology initiative focused on emergency awareness,
              preparedness and community-impacting conditions.
            </p>
            <Link
              href="/earthdial"
              className="mt-6 inline-flex items-center gap-1.5 text-sm font-medium text-foreground transition-colors hover:text-primary"
            >
              Explore EarthDial
              <ArrowRight className="size-4" aria-hidden />
            </Link>
          </Card>
          <Card className="flex flex-col p-8">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
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

      {/* SECTION 8 — Foundation trust */}
      <section className="border-t border-border py-20 sm:py-24">
        <Container className="grid items-center gap-12 lg:grid-cols-2">
          <div className="overflow-hidden rounded-xl border border-primary/25 shadow-2xl shadow-black/40">
            <Image
              src={siteConfig.foundationGraphic}
              alt={siteConfig.foundationGraphicAlt}
              width={1485}
              height={1050}
              className="h-auto w-full"
            />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
              Verified Foundation Status
            </p>
            <h2 className="mt-4 font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
              A Verified Private Foundation Supporting Public-Benefit Technology
            </h2>
            <p className="mt-5 text-base leading-relaxed text-muted-foreground">
              Black Diamond Project Corp is listed in IRS Publication 78 Data as an organization
              eligible to receive tax-deductible charitable contributions. IRS deductibility code:
              PF — Private Foundation.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/foundation-status"
                className={cn(buttonVariants({ variant: "primary", size: "md" }))}
              >
                View Foundation Status
              </Link>
              <Link
                href="/support"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                Support the Mission
              </Link>
            </div>
          </div>
        </Container>
      </section>

      {/* SECTION 9 — Leadership preview */}
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

      {/* SECTION 10 — FAQ */}
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

      {/* SECTION 11 — Support */}
      <SupportSection />
    </>
  )
}
