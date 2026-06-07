import Link from "next/link"
import { cn } from "@/lib/utils"
import { donationDisclosure } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, DiamondMark } from "./primitives"

export function SupportSection() {
  return (
    <section className="border-y border-border bg-[oklch(0.14_0.004_286)] py-20 sm:py-24">
      <Container>
        <div className="grid gap-10 border-l border-primary/45 pl-6 sm:pl-8 lg:grid-cols-[1fr_auto] lg:items-end">
          <div className="max-w-3xl">
          <DiamondMark className="size-8" />
          <h2 className="mt-6 font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
            Support Responsible Technology Research.
          </h2>
          <p className="mt-5 text-lg leading-relaxed text-muted-foreground">
            Support helps Black Diamond Project Corp advance public-benefit research in trustworthy
            AI, quantum resilience, cybersecurity assurance, and public-safety innovation.
          </p>

          <p className="mt-8 max-w-2xl text-xs leading-relaxed text-muted-foreground">
            {donationDisclosure}
          </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
            <Link
              href="/support"
              className={cn(buttonVariants({ variant: "primary", size: "lg" }))}
            >
              Support the Mission
            </Link>
            <Link
              href="/contact"
              className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
            >
              Partner With Us
            </Link>
          </div>

        </div>
      </Container>
    </section>
  )
}
