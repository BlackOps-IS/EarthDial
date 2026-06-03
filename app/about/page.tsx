import type { Metadata } from "next"
import Link from "next/link"
import { Container, SectionHeading } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { buttonVariants } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { siteConfig, missionPrinciples, researchApproach } from "@/lib/content"

export const metadata: Metadata = {
  title: "About",
  description:
    "Black Diamond Project Corp is a California nonprofit corporation and IRS-listed private foundation advancing secure AI, post-quantum cybersecurity, privacy-first systems, and public-safety resilience technology.",
  alternates: { canonical: "/about" },
}

const focusAreas = [
  "Secure and responsible, human-overseen artificial intelligence",
  "Post-quantum cybersecurity and long-term data protection",
  "Privacy-first systems and secure computing, including Reldun OS",
  "Public-safety resilience and emergency-awareness technology",
]

export default function AboutPage() {
  return (
    <>
      <PageHeader
        eyebrow="About"
        title="A private foundation advancing public-benefit technology."
        description={siteConfig.locationNeutral}
      />

      <section className="py-16 sm:py-20">
        <Container className="grid gap-12 lg:grid-cols-[1.4fr_1fr]">
          <div className="flex flex-col gap-6 text-base leading-relaxed text-muted-foreground">
            <p>
              Black Diamond Project Corp is a nonprofit organization dedicated to advancing
              technology for public benefit. Our work focuses on responsible research in
              trustworthy artificial intelligence, quantum resilience, cybersecurity assurance,
              and public-safety innovation.
            </p>
            <p>
              We approach each initiative with a commitment to human oversight, auditability, and
              evidence-grounded design. We describe our programs honestly: what stage they are at,
              what they are intended to support, and the limitations that apply. We do not claim
              deployments, funding, awards, endorsements, or partnerships that have not occurred.
            </p>
            <p>
              The organization is listed in IRS Publication 78 as eligible to receive
              tax-deductible charitable contributions and is classified by the IRS as a private
              foundation. This recognition supports our ability to pursue long-horizon,
              public-benefit research with integrity.
            </p>
          </div>

          <Card className="h-fit p-7">
            <h3 className="font-serif text-lg font-medium">Organizational Profile</h3>
            <dl className="mt-5 flex flex-col gap-4 text-sm">
              <div className="flex flex-col gap-1">
                <dt className="text-muted-foreground">Organization</dt>
                <dd className="font-medium">{siteConfig.organizationName}</dd>
              </div>
              <div className="flex flex-col gap-1">
                <dt className="text-muted-foreground">Entity Type</dt>
                <dd className="font-medium">California Nonprofit Corporation</dd>
              </div>
              <div className="flex flex-col gap-1">
                <dt className="text-muted-foreground">IRS Classification</dt>
                <dd className="font-medium">{siteConfig.foundationClassification}</dd>
              </div>
              <div className="flex flex-col gap-1">
                <dt className="text-muted-foreground">Charitable Eligibility</dt>
                <dd className="font-medium">Listed in IRS Publication 78</dd>
              </div>
              <div className="flex flex-col gap-1">
                <dt className="text-muted-foreground">Contact</dt>
                <dd className="font-medium">
                  <a href={`mailto:${siteConfig.contactEmail}`} className="hover:text-primary">
                    {siteConfig.contactEmail}
                  </a>
                </dd>
              </div>
            </dl>
          </Card>
        </Container>
      </section>

      <section className="border-t border-border py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="What We Pursue"
            title="Focus areas"
            description="Our research is concentrated in domains where trust, resilience, and rigor matter most."
          />
          <ul className="mt-10 grid gap-3 sm:grid-cols-2">
            {focusAreas.map((area) => (
              <li
                key={area}
                className="flex items-start gap-3 rounded-lg border border-border bg-card/40 p-5 text-sm leading-relaxed"
              >
                <span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
                {area}
              </li>
            ))}
          </ul>
        </Container>
      </section>

      <section className="border-t border-border py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="How We Work"
            title="Principles that guide every initiative"
          />
          <div className="mt-10 grid gap-5 sm:grid-cols-3">
            {missionPrinciples.map((p) => (
              <Card key={p.title} className="p-6">
                <h3 className="font-serif text-lg font-medium">{p.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{p.body}</p>
              </Card>
            ))}
          </div>
        </Container>
      </section>

      <section className="border-t border-border bg-diamond-grid py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Our Method"
            title="A disciplined research process"
            description="Each initiative moves through a structured path from mission need to public-benefit translation."
          />
          <ol className="mt-10 grid gap-5 md:grid-cols-5">
            {researchApproach.map((stage) => (
              <li key={stage.step} className="flex flex-col gap-3">
                <span className="font-serif text-2xl font-medium text-primary">{stage.step}</span>
                <h3 className="text-sm font-semibold">{stage.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{stage.body}</p>
              </li>
            ))}
          </ol>

          <div className="mt-12 flex flex-col gap-3 sm:flex-row">
            <Link
              href="/research"
              className={cn(buttonVariants({ variant: "primary", size: "lg" }))}
            >
              Explore Our Research
            </Link>
            <Link
              href="/foundation-status"
              className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
            >
              View Foundation Status
            </Link>
          </div>
        </Container>
      </section>
    </>
  )
}
