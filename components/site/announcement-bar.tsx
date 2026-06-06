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
    <div className="border-b border-primary/25 bg-[oklch(0.145_0.012_84)] text-sm">
      <Container className="flex min-h-9 items-center gap-2 py-2 sm:gap-3">
        <span className="hidden size-5 shrink-0 items-center justify-center rounded-full border border-primary/35 bg-primary/10 text-primary sm:inline-flex">
          <BadgeCheck className="size-3.5" aria-hidden />
        </span>
        <p className="min-w-0 flex-1 truncate text-[0.76rem] leading-snug text-foreground/90 sm:text-[0.8rem]">
          <span className="hidden sm:inline">
            <span className="font-semibold text-primary">Foundation status:</span>{" "}
            Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive
            tax-deductible charitable contributions.
          </span>
          <span className="font-semibold text-primary sm:hidden">IRS Publication 78</span>
        </p>
        <Link
          href={announcement.ctaHref}
          className="group inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-[0.8rem] font-semibold text-primary transition-colors hover:text-foreground"
        >
          <span className="sm:hidden">Verify</span>
          <span className="hidden sm:inline">{announcement.ctaLabel}</span>
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
