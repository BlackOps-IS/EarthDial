import Link from "next/link"
import { footerNav, footerLegal, siteConfig } from "@/lib/content"
import { Container, Logo } from "./primitives"

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-border bg-[oklch(0.14_0.004_286)]">
      <Container className="py-14">
        <div className="flex flex-col gap-10 md:flex-row md:items-start md:justify-between">
          <div className="max-w-sm">
            <Logo />
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
              {siteConfig.tagline}
            </p>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              {siteConfig.locationNeutral}
            </p>
          </div>

          <nav aria-label="Footer" className="grid grid-cols-2 gap-x-12 gap-y-3 sm:grid-cols-4">
            {footerNav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                {item.label}
              </Link>
            ))}
          </nav>
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
