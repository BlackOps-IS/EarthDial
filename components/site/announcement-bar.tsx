"use client"

import { useState } from "react"
import Link from "next/link"
import { ArrowRight, BadgeCheck, X } from "lucide-react"
import { announcement } from "@/lib/content"
import { Container } from "./primitives"

export function AnnouncementBar() {
  const [dismissed, setDismissed] = useState(false)
  if (dismissed) return null

  return (
    <div className="border-b border-primary/25 bg-[linear-gradient(90deg,oklch(0.17_0.025_92),oklch(0.14_0.012_286)_45%,oklch(0.18_0.026_84))] text-sm">
      <Container className="flex min-h-9 items-center gap-3 py-2">
        <span className="hidden size-5 shrink-0 items-center justify-center rounded-full border border-primary/35 bg-primary/10 text-primary sm:inline-flex">
          <BadgeCheck className="size-3.5" aria-hidden />
        </span>
        <p className="flex-1 text-pretty text-[0.8rem] leading-snug text-foreground/90">
          <span className="font-semibold text-primary">Foundation status:</span>{" "}
          <span className="hidden sm:inline">
            Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive
            tax-deductible charitable contributions.
          </span>
          <span className="sm:hidden">Listed in IRS Publication 78 Data.</span>
        </p>
        <Link
          href={announcement.ctaHref}
          className="group inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-[0.8rem] font-semibold text-primary transition-colors hover:text-foreground"
        >
          {announcement.ctaLabel}
          <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden />
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
