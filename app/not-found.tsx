import Link from "next/link"
import { Container, DiamondMark } from "@/components/site/primitives"
import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export default function NotFound() {
  return (
    <section className="bg-diamond-grid py-28 sm:py-36">
      <Container className="flex flex-col items-center text-center">
        <DiamondMark className="size-10" />
        <p className="mt-8 text-xs font-semibold uppercase tracking-[0.22em] text-primary">
          404
        </p>
        <h1 className="mt-4 font-serif text-4xl font-medium tracking-tight text-balance sm:text-5xl">
          Page not found.
        </h1>
        <p className="mt-5 max-w-md text-base leading-relaxed text-muted-foreground">
          The page you are looking for may have moved. Explore our mission and research from the
          links below.
        </p>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <Link href="/" className={cn(buttonVariants({ variant: "primary", size: "lg" }))}>
            Return Home
          </Link>
          <Link
            href="/research"
            className={cn(buttonVariants({ variant: "outline", size: "lg" }))}
          >
            View Research
          </Link>
        </div>
      </Container>
    </section>
  )
}
