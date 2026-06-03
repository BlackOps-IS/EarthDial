"use client"

import { useState } from "react"
import Link from "next/link"
import { ArrowRight, X } from "lucide-react"
import { announcement } from "@/lib/content"
import { Container } from "./primitives"

export function AnnouncementBar() {
  const [dismissed, setDismissed] = useState(false)
  if (dismissed) return null

  return (
    <div className="border-b border-primary/15 bg-[oklch(0.19_0.02_92)] text-sm">
      <Container className="flex items-center gap-3 py-2">
        <p className="flex-1 text-pretty text-[0.8rem] leading-snug text-foreground/90">
          <span className="font-semibold text-primary">Milestone:</span>{" "}
          <span className="hidden sm:inline">
            Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive
            tax-deductible charitable contributions.
          </span>
          <span className="sm:hidden">Listed in IRS Publication 78 Data.</span>
        </p>
        <Link
          href={announcement.ctaHref}
          className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-[0.8rem] font-semibold text-primary hover:underline"
        >
          {announcement.ctaLabel}
          <ArrowRight className="size-3.5" aria-hidden />
        </Link>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          aria-label="Dismiss announcement"
          className="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:text-foreground sm:hidden"
        >
          <X className="size-4" aria-hidden />
        </button>
      </Container>
    </div>
  )
}
