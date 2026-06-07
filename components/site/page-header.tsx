import { Container, Eyebrow } from "./primitives"
import { cn } from "@/lib/utils"

export function PageHeader({
  eyebrow,
  title,
  description,
  children,
  className,
}: {
  eyebrow?: string
  title: string
  description?: string
  children?: React.ReactNode
  className?: string
}) {
  return (
    <section
      className={cn(
        "relative overflow-hidden border-b border-border bg-[oklch(0.14_0.004_286)]",
        className,
      )}
    >
      <Container className="py-14 sm:py-18">
        <div className="max-w-4xl border-l border-primary/45 pl-6 sm:pl-8">
          {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
          <h1 className="mt-4 font-serif text-4xl font-medium leading-[1.08] tracking-tight text-balance sm:text-[3.4rem]">
            {title}
          </h1>
          {description ? (
            <p className="mt-5 text-lg leading-relaxed text-muted-foreground">{description}</p>
          ) : null}
          {children ? <div className="mt-7">{children}</div> : null}
        </div>
      </Container>
    </section>
  )
}
