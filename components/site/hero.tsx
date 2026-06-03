import Link from "next/link"
import Image from "next/image"
import { ArrowRight, ShieldCheck, Atom, Cpu, Radar } from "lucide-react"
import { cn } from "@/lib/utils"
import { siteConfig, heroCredibility } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container, Eyebrow, DiamondMark } from "./primitives"

const pillars = [
  { icon: ShieldCheck, label: "Secure & Responsible AI" },
  { icon: Atom, label: "Post-Quantum Security" },
  { icon: Cpu, label: "Privacy-First Systems" },
  { icon: Radar, label: "Public-Safety Resilience" },
]

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-border">
      {/* Diamond backdrop */}
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        <Image
          src="/images/hero-diamond.png"
          alt=""
          fill
          priority
          sizes="100vw"
          className="object-cover object-right opacity-50 sm:opacity-70 lg:opacity-100 lg:object-[120%_center]"
        />
        {/* Left-to-right legibility wash */}
        <div className="absolute inset-0 bg-gradient-to-r from-background via-background/85 to-background/30 lg:via-background/60 lg:to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-background to-transparent" />
      </div>

      <Container className="relative py-24 lg:py-36">
        <div className="max-w-2xl">
          <Eyebrow>{siteConfig.organizationName}</Eyebrow>
          <h1 className="mt-6 font-serif text-[2.5rem] font-medium leading-[1.05] tracking-tight text-balance sm:text-6xl lg:text-[4.25rem]">
            Secure Technology for a{" "}
            <span className="text-gradient-gold">Safer and More Resilient Future</span>
          </h1>
          <p className="mt-7 max-w-xl text-lg leading-relaxed text-muted-foreground">
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
              href="/support"
              className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
            >
              Support the Mission
            </Link>
          </div>

          <p className="mt-5 text-sm text-muted-foreground">
            Researching with institutions and collaborators?{" "}
            <Link
              href="/contact"
              className="font-medium text-primary underline-offset-4 hover:underline"
            >
              Partner With Us
            </Link>
          </p>

          {/* Pillars */}
          <ul className="mt-12 flex flex-wrap gap-3">
            {pillars.map((pillar) => (
              <li
                key={pillar.label}
                className="inline-flex items-center gap-2.5 rounded-full border border-border bg-card/60 px-4 py-2 backdrop-blur-sm"
              >
                <pillar.icon className="size-4 text-primary" aria-hidden />
                <span className="text-sm font-medium">{pillar.label}</span>
              </li>
            ))}
          </ul>

          {/* Credibility strip */}
          <ul className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-border pt-6">
            {heroCredibility.map((item) => (
              <li key={item} className="flex items-center gap-2 text-sm text-foreground/80">
                <DiamondMark className="size-3.5" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </Container>
    </section>
  )
}
