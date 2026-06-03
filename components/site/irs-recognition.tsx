import Image from "next/image"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { siteConfig } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, Eyebrow } from "./primitives"

export function IRSRecognitionSection() {
  return (
    <section className="border-b border-border py-20 sm:py-24">
      <Container className="grid items-center gap-12 lg:grid-cols-2">
        <div className="order-2 lg:order-1">
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
        </div>

        <div className="order-1 lg:order-2">
          <Eyebrow>Foundation Milestone</Eyebrow>
          <h2 className="mt-5 font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
            Officially Listed in IRS Publication 78
          </h2>
          <p className="mt-5 text-lg leading-relaxed text-muted-foreground">
            Black Diamond Project Corp is listed in IRS Publication 78 as an organization eligible
            to receive tax-deductible charitable contributions.
          </p>
          <p className="mt-4 inline-flex items-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm font-medium">
            <span className="text-muted-foreground">IRS Classification:</span>
            <span className="text-primary">{siteConfig.foundationClassification}</span>
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
  )
}
