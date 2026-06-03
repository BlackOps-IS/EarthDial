import type { Metadata } from "next"
import { Container } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { siteConfig } from "@/lib/content"

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "How Black Diamond Project Corp collects, uses, and protects information submitted through this website.",
}

const sections = [
  {
    heading: "Overview",
    body: "This Privacy Policy explains how Black Diamond Project Corp ('we', 'us', or 'our') handles information collected through this website. We are committed to handling information responsibly and using it only for the purposes described here.",
  },
  {
    heading: "Information We Collect",
    body: "We collect information you choose to provide, such as your name, email address, organization, and message when you submit a contact form. We may also collect standard technical information, such as browser type and pages visited, through routine web analytics.",
  },
  {
    heading: "How We Use Information",
    body: "We use the information you provide to respond to your inquiries, communicate about our programs and research, and improve our website. We do not sell your personal information.",
  },
  {
    heading: "Information Sharing",
    body: "We do not share your personal information with third parties except as necessary to operate this website, comply with legal obligations, or with your consent. Service providers that support our operations are expected to protect your information.",
  },
  {
    heading: "Data Retention",
    body: "We retain information for as long as needed to fulfill the purposes described in this policy, unless a longer retention period is required by law.",
  },
  {
    heading: "Your Choices",
    body: "You may request access to, correction of, or deletion of the personal information you have provided to us by contacting us using the details below.",
  },
  {
    heading: "Changes to This Policy",
    body: "We may update this Privacy Policy from time to time. Material changes will be reflected by updating the effective date below.",
  },
]

export default function PrivacyPage() {
  return (
    <>
      <PageHeader eyebrow="Legal" title="Privacy Policy" />
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
                Questions about this Privacy Policy may be directed to{" "}
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
