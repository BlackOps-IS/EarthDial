import Link from "next/link"
import Image from "next/image"
import { cn } from "@/lib/utils"
import { donationDisclosure } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, DiamondMark } from "./primitives"

export function SupportSection() {
  return (
    <section className="relative overflow-hidden border-y border-border py-24 sm:py-28">
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        <Image
          src="/images/lattice-texture.png"
          alt=""
          fill
          sizes="100vw"
          className="object-cover opacity-60"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background via-background/70 to-background" />
      </div>
      <Container className="relative">
        <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
          <DiamondMark className="size-9" />
          <h2 className="mt-6 font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
            Support Responsible Technology Research.
          </h2>
          <p className="mt-5 text-lg leading-relaxed text-muted-foreground">
            Support helps Black Diamond Project Corp advance public-benefit research in trustworthy
            AI, quantum resilience, cybersecurity assurance, and public-safety innovation.
          </p>

          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
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

          <p className="mt-10 max-w-2xl text-xs leading-relaxed text-muted-foreground">
            {donationDisclosure}
          </p>
        </div>
      </Container>
    </section>
  )
}
