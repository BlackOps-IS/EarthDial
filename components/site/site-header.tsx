"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { primaryNav } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, Logo } from "./primitives"

export function SiteHeader() {
  const pathname = usePathname()
  const [open, setOpen] = useState(false)

  // Close the mobile menu on route change.
  useEffect(() => {
    setOpen(false)
  }, [pathname])

  // Lock body scroll while the mobile menu is open.
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : ""
    return () => {
      document.body.style.overflow = ""
    }
  }, [open])

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href)

  return (
    <header className="sticky top-0 z-50 border-b border-border/80 bg-background/85 backdrop-blur-md">
      <Container className="flex h-16 items-center justify-between gap-4">
        <Link href="/" aria-label="Black Diamond Project Corp home" className="rounded">
          <Logo />
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-7 lg:flex">
          {primaryNav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive(item.href) ? "page" : undefined}
              className={cn(
                "text-sm font-medium transition-colors hover:text-foreground",
                isActive(item.href) ? "text-foreground" : "text-muted-foreground",
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-3 lg:flex">
          <Link href="/contact" className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}>
            Contact Us
          </Link>
          <Link
            href="/support"
            className={cn(buttonVariants({ variant: "primary", size: "sm" }))}
          >
            Support the Mission
          </Link>
        </div>

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls="mobile-nav"
          aria-label={open ? "Close menu" : "Open menu"}
          className="inline-flex size-10 items-center justify-center rounded-md text-foreground lg:hidden"
        >
          {open ? <X className="size-5" aria-hidden /> : <Menu className="size-5" aria-hidden />}
        </button>
      </Container>

      {/* Mobile navigation */}
      <div
        id="mobile-nav"
        className={cn(
          "lg:hidden",
          open ? "block" : "hidden",
        )}
      >
        <nav
          aria-label="Mobile"
          className="border-t border-border bg-background"
        >
          <Container className="flex flex-col gap-1 py-4">
            {primaryNav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive(item.href) ? "page" : undefined}
                className={cn(
                  "rounded-md px-3 py-2.5 text-base font-medium transition-colors",
                  isActive(item.href)
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                {item.label}
              </Link>
            ))}
            <div className="mt-3 flex flex-col gap-2">
              <Link
                href="/contact"
                className={cn(buttonVariants({ variant: "outline", size: "md" }), "w-full")}
              >
                Contact Us
              </Link>
              <Link
                href="/support"
                className={cn(buttonVariants({ variant: "primary", size: "md" }), "w-full")}
              >
                Support the Mission
              </Link>
            </div>
          </Container>
        </nav>
      </div>
    </header>
  )
}
