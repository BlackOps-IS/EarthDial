import Link from "next/link"
import { ArrowRight, ExternalLink, Mail } from "lucide-react"
import { cn } from "@/lib/utils"
import { footerNav, siteConfig } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, DiamondMark, Logo } from "./primitives"

const linkClass =
  "inline-flex min-h-9 items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-primary"

const columns = [footerNav.organization, footerNav.research, footerNav.getInvolved]

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-primary/25 bg-[oklch(0.125_0.004_286)]">
      <div className="border-b border-border bg-diamond-grid">
        <Container className="grid gap-8 py-10 sm:py-12 lg:grid-cols-[1fr_auto] lg:items-center">
          <div className="max-w-2xl">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">
              Work With Black Diamond
            </p>
            <h2 className="mt-3 font-serif text-2xl font-medium leading-tight text-foreground sm:text-3xl">
              Advance technology built for trust, resilience, and public benefit.
            </h2>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row lg:justify-end">
            <Link
              href="/contact"
              className={cn(buttonVariants({ variant: "outline", size: "md" }), "sm:min-w-40")}
            >
              Partner With Us
            </Link>
            <Link
              href="/support"
              className={cn(
                buttonVariants({ variant: "primary", size: "md" }),
                "group sm:min-w-44",
              )}
            >
              Support the Mission
              <ArrowRight
                className="transition-transform group-hover:translate-x-0.5"
                aria-hidden
              />
            </Link>
          </div>
        </Container>
      </div>

      <Container className="py-12 sm:py-14">
        <div className="grid gap-12 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1.85fr)] lg:gap-16">
          <div>
            <Link
              href="/"
              aria-label="Black Diamond Project Corp home"
              className="inline-flex rounded-sm"
            >
              <Logo />
            </Link>
            <p className="mt-5 max-w-md text-sm leading-6 text-muted-foreground">
              {siteConfig.locationNeutral}
            </p>
            <address className="mt-6 not-italic">
              <a
                href={`mailto:${siteConfig.contactEmail}`}
                className="inline-flex min-h-10 items-center gap-2 text-sm font-medium text-foreground transition-colors hover:text-primary"
              >
                <span className="grid size-8 place-items-center border border-primary/25 text-primary">
                  <Mail className="size-4" aria-hidden />
                </span>
                {siteConfig.contactEmail}
              </a>
            </address>
          </div>

          <div className="grid grid-cols-2 gap-x-6 gap-y-10 sm:grid-cols-3 sm:gap-x-10">
            {columns.map((column) => (
              <nav key={column.heading} aria-label={column.heading}>
                <h2 className="border-b border-primary/20 pb-3 text-xs font-semibold uppercase tracking-[0.18em] text-foreground/80">
                  {column.heading}
                </h2>
                <ul className="mt-3 flex flex-col">
                  {column.links.map((item) =>
                    "external" in item && item.external ? (
                      <li key={item.label}>
                        <a
                          href={item.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={linkClass}
                        >
                          {item.label}
                          <ExternalLink className="size-3.5" aria-hidden />
                          <span className="sr-only">(opens in a new tab)</span>
                        </a>
                      </li>
                    ) : (
                      <li key={item.label}>
                        <Link href={item.href} className={linkClass}>
                          {item.label}
                        </Link>
                      </li>
                    ),
                  )}
                </ul>
              </nav>
            ))}
          </div>
        </div>

        <div className="mt-12 flex flex-col gap-5 border-t border-border pt-7 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <DiamondMark className="size-5 shrink-0 text-primary" />
            <p>
              &copy; {new Date().getFullYear()} {siteConfig.organizationName}. All rights reserved.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <Link href="/privacy" className="transition-colors hover:text-primary">
              Privacy
            </Link>
            <Link href="/terms" className="transition-colors hover:text-primary">
              Terms
            </Link>
            <span>bdproj.org</span>
          </div>
        </div>
      </Container>
    </footer>
  )
}
