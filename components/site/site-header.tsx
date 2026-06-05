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
    <header className="sticky top-0 z-50 border-b border-primary/20 bg-background/95 shadow-[0_1px_0_rgb(242_212_122_/_0.08),0_16px_48px_rgb(0_0_0_/_0.32)] backdrop-blur-xl">
      <Container className="flex min-h-[4.75rem] items-center justify-between gap-4">
        <Link
          href="/"
          aria-label="Black Diamond Project Corp home"
          className="shrink-0 rounded-sm py-2"
        >
          <Logo />
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-1 lg:flex">
          {primaryNav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive(item.href) ? "page" : undefined}
              className={cn(
                "whitespace-nowrap rounded-sm px-3 py-2 text-sm font-medium transition-colors hover:bg-muted/70 hover:text-foreground",
                isActive(item.href)
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground",
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-3 lg:flex">
          <Link
            href="/contact"
            className={cn(
              buttonVariants({ variant: "ghost", size: "sm" }),
              "hover:bg-primary/10 hover:text-primary",
            )}
          >
            Partner With Us
          </Link>
          <Link
            href="/support"
            className={cn(
              buttonVariants({ variant: "primary", size: "sm" }),
              "shadow-[0_8px_24px_rgb(242_212_122_/_0.16)]",
            )}
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
          className="inline-flex size-10 items-center justify-center rounded-sm border border-primary/20 text-foreground lg:hidden"
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
          className="border-t border-primary/15 bg-background"
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
                Partner With Us
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
