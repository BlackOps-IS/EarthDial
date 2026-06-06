import Image from "next/image"
import Link from "next/link"
import { Check, ExternalLink, ShieldCheck } from "lucide-react"
import { cn } from "@/lib/utils"
import { siteConfig, donationDisclosure } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Container } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { createPageMetadata } from "@/lib/metadata"

export const metadata = createPageMetadata({
  title: "Foundation Status & IRS Recognition",
  description:
    "Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive tax-deductible charitable contributions. IRS classification: Private Foundation.",
  path: "/foundation-status",
})

const statusRows = [
  { label: "Organization", value: siteConfig.organizationName },
  { label: "IRS Publication 78 Listing", value: "Yes" },
  { label: "IRS Classification", value: "Private Foundation" },
  { label: "Deductibility Code", value: "PF" },
]

export default function FoundationStatusPage() {
  return (
    <>
      <PageHeader
        eyebrow="Transparency & Recognition"
        title="Foundation Status & IRS Recognition"
        description="Black Diamond Project Corp is listed in IRS Publication 78 Data as an organization eligible to receive tax-deductible charitable contributions."
      />

      <section className="py-16 sm:py-20">
        <Container className="grid gap-12 lg:grid-cols-2 lg:items-start">
          {/* Status card */}
          <div className="min-w-0 flex flex-col gap-6">
            <Card className="hairline-top">
              <CardContent className="p-7">
                <div className="flex items-center gap-3">
                  <span className="inline-flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <ShieldCheck className="size-5" aria-hidden />
                  </span>
                  <h2 className="font-serif text-xl font-medium tracking-tight">
                    Recognition Summary
                  </h2>
                </div>
                <dl className="mt-6 flex flex-col divide-y divide-border">
                  {statusRows.map((row) => (
                    <div
                      key={row.label}
                      className="flex items-center justify-between gap-4 py-3.5 first:pt-0 last:pb-0"
                    >
                      <dt className="text-sm text-muted-foreground">{row.label}</dt>
                      <dd className="flex items-center gap-2 text-sm font-semibold">
                        {row.value === "Yes" ? (
                          <Check className="size-4 text-primary" aria-hidden />
                        ) : null}
                        {row.value}
                      </dd>
                    </div>
                  ))}
                </dl>
              </CardContent>
            </Card>

            {/* Verification notice */}
            <Card>
              <CardContent className="p-7">
                <h2 className="font-serif text-lg font-medium tracking-tight">
                  Independent Verification
                </h2>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                  Donors and partners may independently verify organizational eligibility through
                  the IRS Tax Exempt Organization Search tool.
                </p>
                <a
                  href="https://apps.irs.gov/app/eos/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={cn(
                    buttonVariants({ variant: "outline", size: "sm" }),
                    "mt-5 h-auto min-h-9 max-w-full whitespace-normal py-2 text-center",
                  )}
                >
                  IRS Tax Exempt Organization Search
                  <ExternalLink className="size-4" aria-hidden />
                </a>
              </CardContent>
            </Card>
          </div>

          {/* Announcement graphic */}
          <div className="min-w-0 flex flex-col gap-6">
            <div className="overflow-hidden rounded-xl border border-primary/25 shadow-2xl shadow-black/40">
              <Image
                src={siteConfig.foundationGraphic}
                alt={siteConfig.foundationGraphicAlt}
                width={1485}
                height={1050}
                className="h-auto w-full"
                priority
              />
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">{donationDisclosure}</p>
            <div className="flex flex-col gap-3 sm:flex-row">
              <Link
                href="/support"
                className={cn(buttonVariants({ variant: "primary", size: "md" }))}
              >
                Support the Mission
              </Link>
              <Link
                href="/contact"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                Contact Us
              </Link>
            </div>
          </div>
        </Container>
      </section>
    </>
  )
}
