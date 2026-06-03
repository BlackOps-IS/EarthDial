import type { MetadataRoute } from "next"
import { siteConfig } from "@/lib/content"

export default function sitemap(): MetadataRoute.Sitemap {
  const routes = [
    "",
    "/mission",
    "/programs",
    "/programs/earthdial",
    "/programs/ai-qec",
    "/research",
    "/research/pqc-readiness",
    "/foundation-status",
    "/about",
    "/support",
    "/contact",
    "/privacy",
    "/terms",
  ]
  const lastModified = new Date()
  return routes.map((route) => ({
    url: `${siteConfig.url}${route}`,
    lastModified,
    changeFrequency: "monthly",
    priority: route === "" ? 1 : 0.7,
  }))
}
