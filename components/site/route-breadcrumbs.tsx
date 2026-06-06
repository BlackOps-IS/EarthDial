"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ChevronRight } from "lucide-react"
import { siteConfig } from "@/lib/content"
import { Container } from "./primitives"

const routeLabels: Record<string, string> = {
  "/about": "About",
  "/ai-qec": "AI-QEC",
  "/contact": "Contact",
  "/earthdial": "EarthDial",
  "/foundation-status": "Foundation Status",
  "/mission": "Mission & Leadership",
  "/post-quantum-security": "Post-Quantum Security",
  "/privacy": "Privacy Policy",
  "/reldun-os": "Reldun OS",
  "/research": "Research",
  "/support": "Support",
  "/terms": "Terms of Use",
}

const researchRoutes = new Set([
  "/ai-qec",
  "/earthdial",
  "/post-quantum-security",
  "/reldun-os",
])

export function RouteBreadcrumbs() {
  const pathname = usePathname()
  const label = routeLabels[pathname]

  if (!label) return null

  const trail = [
    { label: "Home", href: "/" },
    ...(researchRoutes.has(pathname) ? [{ label: "Research", href: "/research" }] : []),
    { label, href: pathname },
  ]

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: trail.map((crumb, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: crumb.label,
      item: `${siteConfig.url}${crumb.href === "/" ? "" : crumb.href}`,
    })),
  }

  return (
    <div className="border-b border-border/70 bg-[oklch(0.145_0.004_286)]">
      <script
        type="application/ld+json"
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Container className="py-3">
        <nav aria-label="Breadcrumb">
          <ol className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            {trail.map((crumb, index) => {
              const current = index === trail.length - 1
              return (
                <li key={crumb.href} className="flex items-center gap-1.5">
                  {current ? (
                    <span className="font-medium text-foreground" aria-current="page">
                      {crumb.label}
                    </span>
                  ) : (
                    <>
                      <Link
                        href={crumb.href}
                        className="rounded-sm transition-colors hover:text-foreground"
                      >
                        {crumb.label}
                      </Link>
                      <ChevronRight className="size-3" aria-hidden />
                    </>
                  )}
                </li>
              )
            })}
          </ol>
        </nav>
      </Container>
    </div>
  )
}
