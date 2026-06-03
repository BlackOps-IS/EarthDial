import type { Metadata } from "next"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { buttonVariants } from "@/components/ui/button"
import { Container, SectionHeading } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { MissionPrinciples } from "@/components/site/mission-principles"
import { ResearchApproachTimeline } from "@/components/site/research-approach"

export const metadata: Metadata = {
  title: "Mission",
  description:
    "Black Diamond Project Corp advances trustworthy AI, quantum-resilient systems, and public-safety technology that strengthens communities and earns trust through responsible design.",
}

export default function MissionPage() {
  return (
    <>
      <PageHeader
        eyebrow="Our Mission"
        title="Advanced Technology With a Public Purpose."
        description="We believe powerful technology should strengthen communities, improve resilience, and earn trust through responsible design. Our work explores how artificial intelligence, quantum-resilient systems, and secure decision-support technologies can serve public safety and scientific progress."
      />

      <section className="py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Guiding Principles"
            title="How We Build"
            description="Three principles shape every initiative we pursue."
          />
          <div className="mt-12">
            <MissionPrinciples />
          </div>
        </Container>
      </section>

      <section className="border-t border-border bg-[oklch(0.14_0.004_286)] py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Our Approach"
            title="Research That Must Earn Trust."
            description="Black Diamond approaches emerging technology with disciplined validation — identifying the need, defining the risk, designing the system, testing the assumptions, documenting limitations, and only then describing what the work can support."
          />
          <div className="mt-12">
            <ResearchApproachTimeline />
          </div>
        </Container>
      </section>

      <section className="py-16 sm:py-20">
        <Container>
          <div className="rounded-xl border border-border bg-card p-8 sm:p-12">
            <h2 className="font-serif text-2xl font-medium tracking-tight text-balance sm:text-3xl">
              A public-benefit purpose
            </h2>
            <p className="mt-5 max-w-3xl text-base leading-relaxed text-muted-foreground">
              {"Black Diamond Project Corp is a California nonprofit corporation supporting public-benefit technology initiatives in the United States. Our research focuses on trustworthy AI, quantum resilience, cybersecurity assurance, and public-safety technology — fields where responsible design and rigorous validation matter most."}
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/programs"
                className={cn(buttonVariants({ variant: "primary", size: "md" }))}
              >
                Explore Our Programs
              </Link>
              <Link
                href="/foundation-status"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                View Foundation Status
              </Link>
            </div>
          </div>
        </Container>
      </section>
    </>
  )
}
