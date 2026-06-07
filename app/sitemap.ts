import type { MetadataRoute } from "next"
import { siteConfig } from "@/lib/content"

export const dynamic = "force-static"

export default function sitemap(): MetadataRoute.Sitemap {
  const routes = [
    "",
    "/mission",
    "/research",
    "/reldun-os",
    "/earthdial",
    "/ai-qec",
    "/post-quantum-security",
    "/foundation-status",
    "/about",
    "/support",
    "/contact",
    "/privacy",
    "/terms",
  ]
  const lastModified = new Date()
  const priorities: Record<string, number> = {
    "": 1,
    "/research": 0.9,
    "/reldun-os": 0.9,
    "/earthdial": 0.9,
    "/post-quantum-security": 0.8,
    "/ai-qec": 0.8,
    "/foundation-status": 0.8,
  }
  return routes.map((route) => ({
    url: `${siteConfig.url}${route}`,
    lastModified,
    changeFrequency: "monthly",
    priority: priorities[route] ?? 0.6,
  }))
}
