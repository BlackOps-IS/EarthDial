import type { Metadata } from "next"
import { Container } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { ContactForm } from "@/components/site/contact-form"
import { Card } from "@/components/ui/card"
import { siteConfig } from "@/lib/content"

export const metadata: Metadata = {
  title: "Contact",
  description:
    "Contact Black Diamond Project Corp for research collaboration, institutional inquiries, public-safety technology questions, or support.",
}

export default function ContactPage() {
  return (
    <>
      <PageHeader
        eyebrow="Contact"
        title="Get in touch with our team."
        description="Whether you are interested in research collaboration, institutional engagement, or supporting our mission, we would like to hear from you."
      />

      <section className="py-16 sm:py-20">
        <Container className="grid gap-12 lg:grid-cols-[1.5fr_1fr]">
          <Card className="p-7 sm:p-9">
            <ContactForm />
          </Card>

          <div className="flex flex-col gap-6">
            <Card className="p-7">
              <h3 className="font-serif text-lg font-medium">Direct Contact</h3>
              <dl className="mt-5 flex flex-col gap-4 text-sm">
                <div className="flex flex-col gap-1">
                  <dt className="text-muted-foreground">Email</dt>
                  <dd>
                    <a
                      href={`mailto:${siteConfig.contactEmail}`}
                      className="font-medium hover:text-primary"
                    >
                      {siteConfig.contactEmail}
                    </a>
                  </dd>
                </div>
                <div className="flex flex-col gap-1">
                  <dt className="text-muted-foreground">Website</dt>
                  <dd className="font-medium">www.bdproj.com</dd>
                </div>
                <div className="flex flex-col gap-1">
                  <dt className="text-muted-foreground">Organization</dt>
                  <dd className="font-medium">{siteConfig.organizationName}</dd>
                </div>
              </dl>
            </Card>

            <Card className="p-7">
              <h3 className="font-serif text-lg font-medium">About Our Work</h3>
              <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                {siteConfig.locationNeutral} We focus on trustworthy AI, quantum resilience,
                cybersecurity assurance, and public-safety technology.
              </p>
            </Card>
          </div>
        </Container>
      </section>
    </>
  )
}
