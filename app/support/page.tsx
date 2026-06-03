import type { Metadata } from "next"
import Link from "next/link"
import { Container, SectionHeading } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { Card } from "@/components/ui/card"
import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { siteConfig, donationDisclosure } from "@/lib/content"

export const metadata: Metadata = {
  title: "Support the Mission",
  description:
    "Support Black Diamond Project Corp, an IRS Publication 78-listed private foundation, in advancing secure AI, post-quantum cybersecurity, privacy-first systems, and public-safety resilience research.",
  alternates: { canonical: "/support" },
}

const supportImpact = [
  {
    title: "Secure & Responsible AI",
    body: "Advancing human-overseen, auditable, and evidence-grounded artificial intelligence.",
  },
  {
    title: "Post-Quantum Cybersecurity",
    body: "Researching quantum-resilient cryptography and long-term protection of sensitive information.",
  },
  {
    title: "Privacy-First Systems",
    body: "Exploring secure computing foundations and kernel-boundary control, including Reldun OS.",
  },
  {
    title: "Public-Safety Resilience",
    body: "Designing resilient, auditable decision support for emergency awareness and response.",
  },
]

export default function SupportPage() {
  return (
    <>
      <PageHeader
        eyebrow="Support"
        title="Support responsible technology research."
        description="Contributions help Black Diamond Project Corp advance public-benefit research in secure AI, post-quantum cybersecurity, privacy-first systems, and public-safety resilience technology."
      />

      <section className="py-16 sm:py-20">
        <Container className="grid gap-12 lg:grid-cols-[1.3fr_1fr]">
          <div className="flex flex-col gap-6">
            <SectionHeading
              eyebrow="Your Impact"
              title="Where support goes"
            />
            <div className="grid gap-5">
              {supportImpact.map((item) => (
                <Card key={item.title} className="p-6">
                  <h3 className="font-serif text-lg font-medium">{item.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{item.body}</p>
                </Card>
              ))}
            </div>
          </div>

          <Card className="flex h-fit flex-col gap-5 p-7">
            <h3 className="font-serif text-xl font-medium">Make a Contribution</h3>
            {siteConfig.donationEnabled ? (
              <Link
                href="/contact"
                className={cn(buttonVariants({ variant: "primary", size: "lg" }))}
              >
                Continue to Giving
              </Link>
            ) : (
              <>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Online giving is being prepared. To discuss a contribution or institutional
                  support in the meantime, please contact us directly and our team will follow up.
                </p>
                <Link
                  href="/contact"
                  className={cn(buttonVariants({ variant: "primary", size: "lg" }))}
                >
                  Contact Us to Give
                </Link>
                <a
                  href={`mailto:${siteConfig.contactEmail}`}
                  className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
                >
                  Email {siteConfig.contactEmail}
                </a>
              </>
            )}
            <p className="border-t border-border pt-5 text-xs leading-relaxed text-muted-foreground">
              {donationDisclosure}
            </p>
          </Card>
        </Container>
      </section>
    </>
  )
}
