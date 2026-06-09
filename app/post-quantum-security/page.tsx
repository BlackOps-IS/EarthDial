import Link from "next/link"
import { Check } from "lucide-react"
import { cn } from "@/lib/utils"
import { pqcSections } from "@/lib/content"
import { buttonVariants } from "@/components/ui/button"
import { Container } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { createPageMetadata } from "@/lib/metadata"

export const metadata = createPageMetadata({
  title: "Post-Quantum Cybersecurity Research",
  description:
    "Explore Black Diamond Project Corp research into quantum-resilient security and future-ready protection of sensitive information.",
  path: "/post-quantum-security",
})

export default function PostQuantumSecurityPage() {
  return (
    <>
      <PageHeader
        eyebrow="Post-Quantum Cybersecurity"
        title="Preparing Security for the Post-Quantum Future"
        description="A practical, educational overview of how organizations can approach cryptographic readiness in a changing standards environment. This is an educational resource, not a commercial service."
      />

      <section className="py-16 sm:py-20">
        <Container className="max-w-3xl">
          <p className="text-lg leading-relaxed text-muted-foreground">
            {"Advances in quantum computing have prompted standards bodies and organizations to begin preparing for a transition to post-quantum cryptography. The work is less about a single switch and more about disciplined preparation: understanding what you have, where the risk lies, and how to migrate without disruption. The four areas below outline a pragmatic readiness path."}
          </p>

          <div className="mt-14 flex flex-col gap-12">
            {pqcSections.map((section, i) => (
              <article key={section.title} className="border-t border-border pt-8 first:border-t-0 first:pt-0">
                <div className="flex items-baseline gap-3">
                  <span className="font-serif text-xl font-medium text-primary">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <h2 className="font-serif text-2xl font-medium tracking-tight">{section.title}</h2>
                </div>
                <p className="mt-4 text-base leading-relaxed text-muted-foreground">{section.body}</p>
                {section.points ? (
                  <ul className="mt-5 grid gap-2.5">
                    {section.points.map((point) => (
                      <li key={point} className="flex items-start gap-2.5 text-sm leading-relaxed">
                        <Check className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
                        <span className="text-foreground/90">{point}</span>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </article>
            ))}
          </div>

          <div className="mt-14 rounded-lg border border-border bg-card p-6 sm:p-8">
            <h2 className="font-serif text-xl font-medium tracking-tight">
              Research in service of resilience
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              {"Post-quantum readiness is part of Black Diamond's broader research into quantum resilience and cybersecurity assurance. To discuss collaboration or learn more about our work, reach out."}
            </p>
            <div className="mt-6 flex flex-col gap-3 sm:flex-row">
              <Link href="/contact" className={cn(buttonVariants({ variant: "primary", size: "md" }))}>
                Discuss This Research
              </Link>
              <Link
                href="/ai-qec"
                className={cn(buttonVariants({ variant: "outline", size: "md" }))}
              >
                Explore AI-QEC
              </Link>
            </div>
          </div>
        </Container>
      </section>
    </>
  )
}
