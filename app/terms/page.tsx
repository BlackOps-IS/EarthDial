import type { Metadata } from "next"
import { Container } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { siteConfig } from "@/lib/content"

export const metadata: Metadata = {
  title: "Terms of Use",
  description:
    "The terms that govern your use of the Black Diamond Project Corp website and its content.",
  alternates: { canonical: "/terms" },
}

const sections = [
  {
    heading: "Acceptance of Terms",
    body: "By accessing or using this website, you agree to be bound by these Terms of Use. If you do not agree, please do not use this website.",
  },
  {
    heading: "Informational Purpose",
    body: "The content on this website is provided for general informational purposes about Black Diamond Project Corp and its research initiatives. Descriptions of programs reflect their current stage, which may be conceptual, proposed, or in research. Nothing on this website constitutes a representation of deployment, funding, award, endorsement, or partnership unless explicitly and accurately stated.",
  },
  {
    heading: "No Professional Advice",
    body: "Educational resources, including materials on post-quantum readiness and cybersecurity, are provided for general information only and do not constitute legal, financial, security, or other professional advice. Consult a qualified professional regarding your specific circumstances.",
  },
  {
    heading: "Intellectual Property",
    body: "The content, design, and materials on this website are the property of Black Diamond Project Corp or its licensors and are protected by applicable intellectual property laws. You may not reproduce or distribute content without permission, except as permitted by law.",
  },
  {
    heading: "Charitable Contributions",
    body: "Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive tax-deductible charitable contributions and is classified by the IRS as a private foundation. The deductibility of any contribution depends on your individual circumstances; consult your tax adviser.",
  },
  {
    heading: "Limitation of Liability",
    body: "This website is provided on an 'as is' basis. To the fullest extent permitted by law, Black Diamond Project Corp disclaims liability for any damages arising from your use of, or inability to use, this website or its content.",
  },
  {
    heading: "Changes to These Terms",
    body: "We may revise these Terms of Use from time to time. Continued use of the website after changes are posted constitutes acceptance of the revised terms.",
  },
]

export default function TermsPage() {
  return (
    <>
      <PageHeader eyebrow="Legal" title="Terms of Use" />
      <section className="py-16 sm:py-20">
        <Container className="max-w-3xl">
          <p className="text-sm text-muted-foreground">Effective date: January 2025</p>
          <div className="mt-10 flex flex-col gap-10">
            {sections.map((s) => (
              <div key={s.heading} className="flex flex-col gap-3">
                <h2 className="font-serif text-xl font-medium">{s.heading}</h2>
                <p className="text-base leading-relaxed text-muted-foreground">{s.body}</p>
              </div>
            ))}
            <div className="flex flex-col gap-3 border-t border-border pt-10">
              <h2 className="font-serif text-xl font-medium">Contact</h2>
              <p className="text-base leading-relaxed text-muted-foreground">
                Questions about these Terms of Use may be directed to{" "}
                <a
                  href={`mailto:${siteConfig.contactEmail}`}
                  className="text-primary hover:underline"
                >
                  {siteConfig.contactEmail}
                </a>
                .
              </p>
            </div>
          </div>
        </Container>
      </section>
    </>
  )
}
