import Link from "next/link"
import { ChevronRight } from "lucide-react"
import { siteConfig } from "@/lib/content"

type Crumb = { label: string; href: string }

export function Breadcrumbs({ items }: { items: Crumb[] }) {
  const trail = [{ label: "Home", href: "/" }, ...items]

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: trail.map((c, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: c.label,
      item: `${siteConfig.url}${c.href === "/" ? "" : c.href}`,
    })),
  }

  return (
    <nav aria-label="Breadcrumb" className="text-sm">
      <script
        type="application/ld+json"
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <ol className="flex flex-wrap items-center gap-1.5 text-muted-foreground">
        {trail.map((crumb, i) => {
          const last = i === trail.length - 1
          return (
            <li key={crumb.href} className="flex items-center gap-1.5">
              {last ? (
                <span className="font-medium text-foreground" aria-current="page">
                  {crumb.label}
                </span>
              ) : (
                <>
                  <Link href={crumb.href} className="transition-colors hover:text-foreground">
                    {crumb.label}
                  </Link>
                  <ChevronRight className="size-3.5" aria-hidden />
                </>
              )}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
