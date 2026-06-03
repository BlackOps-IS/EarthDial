import Link from "next/link"
import { ExternalLink } from "lucide-react"
import { footerNav, footerLegal, siteConfig } from "@/lib/content"
import { Container, Logo } from "./primitives"

const linkClass =
  "inline-flex min-h-9 items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"

const columns = [footerNav.organization, footerNav.research, footerNav.support]

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-border bg-[oklch(0.14_0.004_286)]">
      <Container className="py-14">
        <div className="grid gap-10 md:grid-cols-12">
          <div className="md:col-span-4 lg:col-span-5">
            <Logo />
            <p className="mt-4 max-w-sm text-sm leading-relaxed text-muted-foreground">
              {siteConfig.locationNeutral}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 md:col-span-8 lg:col-span-7">
            {columns.map((column) => (
              <nav key={column.heading} aria-label={column.heading}>
                <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground/70">
                  {column.heading}
                </h2>
                <ul className="mt-3 flex flex-col gap-0.5">
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

        <div className="mt-12 border-t border-border pt-8">
          <p className="max-w-3xl text-xs leading-relaxed text-muted-foreground">{footerLegal}</p>
          <div className="mt-4 flex flex-col gap-2 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <p>
              &copy; {new Date().getFullYear()} {siteConfig.organizationName}. All rights reserved.
            </p>
            <a
              href={`mailto:${siteConfig.contactEmail}`}
              className="text-primary transition-colors hover:underline"
            >
              {siteConfig.contactEmail}
            </a>
          </div>
        </div>
      </Container>
    </footer>
  )
}
