import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { siteConfig } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, DiamondMark, Eyebrow } from "./primitives"

const pillars = [
  "Secure & Responsible AI",
  "Post-Quantum Security",
  "Privacy-First Systems",
  "Public-Safety Resilience",
]

export function Hero() {
  return (
    <section className="border-b border-border">
      <Container className="grid gap-14 py-20 lg:grid-cols-[minmax(0,1.6fr)_minmax(18rem,0.65fr)] lg:gap-20 lg:py-28">
        <div className="max-w-3xl">
          <Eyebrow>{siteConfig.organizationName}</Eyebrow>
          <h1 className="mt-6 max-w-full break-words font-serif text-[2.55rem] font-medium leading-[1.02] tracking-tight text-balance sm:text-6xl lg:text-[4.5rem]">
            Secure technology for a safer and more resilient future.
          </h1>
          <p className="mt-7 max-w-xl break-words text-lg leading-relaxed text-muted-foreground">
            {siteConfig.organizationName} advances privacy-first artificial intelligence,
            post-quantum cybersecurity, secure systems research and public-safety resilience
            technology. Our work is built for environments where trust, privacy and reliability
            matter most.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link
              href="/research"
              className={cn(buttonVariants({ variant: "primary", size: "lg" }))}
            >
              Explore Our Research
              <ArrowRight className="size-4" aria-hidden />
            </Link>
            <Link
              href="/contact"
              className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
            >
              Partner With Us
            </Link>
          </div>
        </div>
        <aside className="border-t border-primary/55 pt-6 lg:mt-8" aria-label="Research focus">
          <div className="flex items-center gap-3">
            <DiamondMark className="size-7" />
            <p className="text-sm font-medium text-foreground">Research focus</p>
          </div>
          <ol className="mt-6 divide-y divide-border border-b border-border">
            {pillars.map((pillar, index) => (
              <li key={pillar} className="grid grid-cols-[2rem_1fr] gap-3 py-4 text-sm">
                <span className="font-serif text-primary">0{index + 1}</span>
                <span className="font-medium leading-snug">{pillar}</span>
              </li>
            ))}
          </ol>
          <p className="mt-6 text-sm leading-relaxed text-muted-foreground">
            Independent research conducted for public benefit, with clear statements of scope,
            status, and limitations.
          </p>
        </aside>
      </Container>
    </section>
  )
}
