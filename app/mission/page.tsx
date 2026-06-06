import type { Metadata } from "next"
import Link from "next/link"
import { ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { leadership, siteConfig } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, SectionHeading } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { MissionPrinciples } from "@/components/site/mission-principles"
import { ResearchApproachTimeline } from "@/components/site/research-approach"

export const metadata: Metadata = {
  title: "Mission & Leadership",
  description:
    "Black Diamond Project Corp advances secure AI, post-quantum cybersecurity, privacy-first systems, and public-safety resilience technology through responsible, evidence-grounded research.",
  alternates: { canonical: "/mission" },
}

const leadershipJsonLd = {
  "@context": "https://schema.org",
  "@type": "ProfilePage",
  about: leadership.map((person) => ({
    "@type": "Person",
    name: person.name,
    jobTitle: person.role,
    description: person.bio,
    worksFor: {
      "@type": "Organization",
      name: siteConfig.organizationName,
      url: siteConfig.url,
    },
    ...(person.credentials
      ? {
          hasCredential: person.credentials.map((credential) => ({
            "@type": "EducationalOccupationalCredential",
            name: credential,
          })),
        }
      : {}),
    sameAs: person.links?.map((link) => link.href) ?? [],
  })),
}

export default function MissionPage() {
  return (
    <>
      <script
        type="application/ld+json"
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: JSON.stringify(leadershipJsonLd) }}
      />
      <PageHeader
        eyebrow="Our Mission"
        title="Advanced Technology With a Public Purpose."
        description="We believe powerful technology should strengthen communities, improve resilience, and earn trust through responsible design. Our work explores how secure AI, post-quantum cybersecurity, privacy-first systems, and public-safety resilience technology can serve a safer future."
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

      {/* Leadership */}
      <section id="leadership" className="scroll-mt-24 py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Leadership"
            title="The People Behind the Work"
            description="Black Diamond Project Corp is guided by researchers committed to disciplined, evidence-grounded innovation in service of the public benefit."
          />
          <div className="mt-12 grid gap-6 md:grid-cols-2">
            {leadership.map((person) => (
              <div key={person.name} className="rounded-xl border border-border bg-card p-8">
                <h3 className="font-serif text-xl font-medium tracking-tight">{person.name}</h3>
                <p className="mt-1 text-sm font-semibold uppercase tracking-[0.14em] text-primary">
                  {person.role}
                </p>
                <p className="mt-5 text-sm leading-relaxed text-muted-foreground">{person.bio}</p>
                {person.credentials ? (
                  <ul className="mt-5 flex flex-col gap-2 border-t border-border pt-5">
                    {person.credentials.map((credential) => (
                      <li
                        key={credential}
                        className="text-sm leading-relaxed text-muted-foreground"
                      >
                        {credential}
                      </li>
                    ))}
                  </ul>
                ) : null}
                {person.links ? (
                  <div className="mt-5 flex flex-wrap gap-2 border-t border-border pt-5">
                    {person.links.map((link) => (
                      <a
                        key={link.href}
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex min-h-9 items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs font-semibold text-foreground transition-colors hover:border-primary/60 hover:text-primary"
                      >
                        {link.label}
                        <ExternalLink className="size-3" aria-hidden />
                        <span className="sr-only">(opens in a new tab)</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </Container>
      </section>

      <section className="border-t border-border py-16 sm:py-20">
        <Container>
          <div className="rounded-xl border border-border bg-card p-8 sm:p-12">
            <h2 className="font-serif text-2xl font-medium tracking-tight text-balance sm:text-3xl">
              A public-benefit purpose
            </h2>
            <p className="mt-5 max-w-3xl text-base leading-relaxed text-muted-foreground">
              {siteConfig.locationNeutral} Our research focuses on secure and responsible AI, post-quantum
              cybersecurity, privacy-first systems, and public-safety resilience — fields where
              responsible design and rigorous validation matter most.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/research"
                className={cn(buttonVariants({ variant: "primary", size: "md" }))}
              >
                Explore Our Research
              </Link>
              <Link
                href="/foundation-status"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                View Foundation Status
              </Link>
            </div>
            <p className="mt-6 text-xs leading-relaxed text-muted-foreground">
              Questions about partnership or collaboration?{" "}
              <Link href="/contact" className="font-medium text-primary hover:underline">
                Contact us at {siteConfig.contactEmail}
              </Link>
              .
            </p>
          </div>
        </Container>
      </section>
    </>
  )
}
