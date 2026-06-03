import Link from "next/link"
import { ArrowLeft, Check, Info, TriangleAlert } from "lucide-react"
import { cn } from "@/lib/utils"
import { type ProgramDetail } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container } from "./primitives"
import { PageHeader } from "./page-header"
import { ProgramStatusPanel } from "./program-status-panel"
import { Breadcrumbs } from "./breadcrumbs"

export function ProgramDetailView({ detail }: { detail: ProgramDetail }) {
  return (
    <>
      <PageHeader eyebrow={detail.eyebrow} title={detail.name} description={detail.subtitle}>
        <div className="flex flex-col gap-4">
          <Breadcrumbs
            items={[
              { label: "Research", href: "/research" },
              { label: detail.name, href: `/${detail.slug}` },
            ]}
          />
          <Link
            href={detail.backHref}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            <ArrowLeft className="size-4" aria-hidden />
            {detail.backLabel}
          </Link>
        </div>
      </PageHeader>

      <section className="py-16 sm:py-20">
        <Container className="grid gap-12 lg:grid-cols-12">
          {/* Status panel + approved message */}
          <aside className="lg:col-span-4 lg:order-2">
            <div className="flex flex-col gap-6 lg:sticky lg:top-24">
              <ProgramStatusPanel items={detail.panel} />
              <div className="rounded-lg border border-primary/25 bg-primary/5 p-5">
                <p className="text-xs font-semibold uppercase tracking-wide text-primary">
                  Approved Statement
                </p>
                <p className="mt-3 text-sm leading-relaxed text-foreground/90">
                  {detail.approvedMessage}
                </p>
              </div>
            </div>
          </aside>

          {/* Long-form sections */}
          <div className="flex flex-col gap-10 lg:col-span-8 lg:order-1">
            {detail.sections.map((section) => (
              <div key={section.title} className="border-t border-border pt-8 first:border-t-0 first:pt-0">
                <h2 className="font-serif text-2xl font-medium tracking-tight">{section.title}</h2>
                <p className="mt-4 text-base leading-relaxed text-muted-foreground">
                  {section.body}
                </p>
                {section.points ? (
                  <ul className="mt-5 grid gap-2.5 sm:grid-cols-2">
                    {section.points.map((point) => (
                      <li
                        key={point}
                        className="flex items-start gap-2.5 rounded-md border border-border bg-card px-4 py-3 text-sm"
                      >
                        <Check className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            ))}

            {/* Safety notice (e.g. EarthDial) */}
            {detail.safetyNotice ? (
              <div className="flex items-start gap-3 rounded-lg border border-primary/30 bg-primary/5 p-5">
                <TriangleAlert className="mt-0.5 size-5 shrink-0 text-primary" aria-hidden />
                <p className="text-sm font-medium leading-relaxed text-foreground/90">
                  {detail.safetyNotice}
                </p>
              </div>
            ) : null}

            {/* Responsible disclosure */}
            <div className="flex items-start gap-3 rounded-lg border border-border bg-muted/40 p-5">
              <Info className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
              <p className="text-sm leading-relaxed text-muted-foreground">{detail.disclosure}</p>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Link
                href="/contact"
                className={cn(buttonVariants({ variant: "primary", size: "md" }))}
              >
                Discuss This Research
              </Link>
              <Link
                href="/research"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                View All Research
              </Link>
            </div>
          </div>
        </Container>
      </section>
    </>
  )
}
