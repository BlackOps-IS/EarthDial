import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { programs } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, SectionHeading } from "@/components/site/primitives"
import { Hero } from "@/components/site/hero"
import { IRSRecognitionSection } from "@/components/site/irs-recognition"
import { MissionPrinciples } from "@/components/site/mission-principles"
import { ProgramCard } from "@/components/site/program-card"
import { ResearchApproachTimeline } from "@/components/site/research-approach"
import { SupportSection } from "@/components/site/support-section"
import { Card, CardContent } from "@/components/ui/card"

export default function HomePage() {
  const featured = programs.filter((p) => p.featured)
  const supporting = programs.find((p) => !p.featured)

  return (
    <>
      {/* SECTION 1 — Hero */}
      <Hero />

      {/* SECTION 2 — IRS / Foundation milestone */}
      <IRSRecognitionSection />

      {/* SECTION 3 — Mission */}
      <section className="py-20 sm:py-24">
        <Container>
          <SectionHeading
            eyebrow="Our Mission"
            title="Advanced Technology With a Public Purpose."
            description="We believe powerful technology should strengthen communities, improve resilience, and earn trust through responsible design. Our work explores how artificial intelligence, quantum-resilient systems, and secure decision-support technologies can serve public safety and scientific progress."
          />
          <div className="mt-12">
            <MissionPrinciples />
          </div>
        </Container>
      </section>

      {/* SECTION 4 — Flagship programs */}
      <section className="border-t border-border bg-[oklch(0.14_0.004_286)] py-20 sm:py-24">
        <Container>
          <SectionHeading
            eyebrow="Programs"
            title="Flagship Research Initiatives"
            description="Two flagship initiatives, supported by a shared research foundation in AI security and quantum readiness."
          />
          <div className="mt-12 grid gap-6 lg:grid-cols-2">
            {featured.map((program) => (
              <ProgramCard key={program.slug} program={program} featured />
            ))}
          </div>
          {supporting ? (
            <div className="mt-6">
              <ProgramCard program={supporting} />
            </div>
          ) : null}
        </Container>
      </section>

      {/* SECTION 5 — Research approach */}
      <section className="py-20 sm:py-24">
        <Container>
          <SectionHeading
            eyebrow="Our Approach"
            title="Research That Must Earn Trust."
            description="Black Diamond approaches emerging technology with disciplined validation: identify the need, define the risk, design the system, test the assumptions, document limitations, and only then describe what the work can support."
          />
          <div className="mt-12">
            <ResearchApproachTimeline />
          </div>
        </Container>
      </section>

      {/* SECTION 6 — Featured insight */}
      <section className="border-t border-border py-20 sm:py-24">
        <Container>
          <Card className="hairline-top overflow-hidden">
            <CardContent className="grid gap-8 p-8 sm:p-12 lg:grid-cols-12 lg:items-center">
              <div className="lg:col-span-8">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
                  Featured Insight
                </p>
                <h2 className="mt-4 font-serif text-3xl font-medium leading-tight tracking-tight text-balance">
                  Preparing for the Post-Quantum Transition
                </h2>
                <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted-foreground">
                  Explore practical considerations for cryptographic inventory, quantum-risk
                  assessment, crypto-agility, and migration planning in a changing standards
                  environment.
                </p>
                <Link
                  href="/research/pqc-readiness"
                  className={cn(buttonVariants({ variant: "primary", size: "md" }), "mt-7")}
                >
                  Read the PQC Readiness Guide
                  <ArrowRight className="size-4" aria-hidden />
                </Link>
              </div>
              <div className="lg:col-span-4">
                <ul className="grid gap-2.5 text-sm">
                  {[
                    "Cryptographic Inventory",
                    "Quantum-Risk Assessment",
                    "Crypto-Agility",
                    "Migration Planning",
                  ].map((item) => (
                    <li
                      key={item}
                      className="rounded-md border border-border bg-background/50 px-4 py-3 font-medium"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>
        </Container>
      </section>

      {/* SECTION 7 — About / leadership preview */}
      <section className="border-t border-border bg-[oklch(0.14_0.004_286)] py-20 sm:py-24">
        <Container>
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <SectionHeading
              eyebrow="About"
              title="Led by Research and Responsibility."
              description="Black Diamond Project Corp was founded to advance public-benefit technology at the intersection of artificial intelligence, cybersecurity, quantum resilience, and emergency-awareness systems."
            />
            <div className="flex flex-col items-start gap-6 lg:items-end">
              <p className="text-sm leading-relaxed text-muted-foreground lg:text-right">
                Founded by Simon Carreras, AI and security researcher.
              </p>
              <Link
                href="/about"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                About Black Diamond
                <ArrowRight className="size-4" aria-hidden />
              </Link>
            </div>
          </div>
        </Container>
      </section>

      {/* SECTION 8 — Support */}
      <SupportSection />
    </>
  )
}
