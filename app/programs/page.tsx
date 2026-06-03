import type { Metadata } from "next"
import { programs } from "@/lib/content"
import { Container } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { ProgramCard } from "@/components/site/program-card"
import { StatusBadge } from "@/components/site/status-badge"

export const metadata: Metadata = {
  title: "Research Initiatives",
  description:
    "Flagship research initiatives from Black Diamond Project Corp: EarthDial Guardian Mesh, AI-Integrated Quantum Error Correction, and supporting AI Security & Quantum Readiness research.",
}

const legend = [
  {
    tone: "gold" as const,
    label: "Submitted Concept",
    body: "A concept submitted as an unclassified RFI response. Not selected, funded, or deployed.",
  },
  {
    tone: "blue" as const,
    label: "Proposed Research",
    body: "A proposed feasibility initiative. No award, funding, or partnership is claimed.",
  },
  {
    tone: "muted" as const,
    label: "Supporting Capability",
    body: "A research foundation supporting flagship initiatives. Not a commercial service.",
  },
]

export default function ProgramsPage() {
  const featured = programs.filter((p) => p.featured)
  const supporting = programs.filter((p) => !p.featured)

  return (
    <>
      <PageHeader
        eyebrow="Programs"
        title="Research Initiatives Designed for Public Benefit."
        description="Black Diamond pursues two flagship research initiatives, supported by a shared foundation in AI security and quantum readiness. Each initiative is presented with a clear, honest status."
      />

      {/* Status legend */}
      <section className="border-b border-border py-10">
        <Container>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            Status Legend
          </p>
          <div className="mt-5 grid gap-4 sm:grid-cols-3">
            {legend.map((item) => (
              <div
                key={item.label}
                className="flex flex-col gap-3 rounded-lg border border-border bg-card p-5"
              >
                <StatusBadge tone={item.tone}>{item.label}</StatusBadge>
                <p className="text-sm leading-relaxed text-muted-foreground">{item.body}</p>
              </div>
            ))}
          </div>
        </Container>
      </section>

      <section className="py-16 sm:py-20">
        <Container>
          <div className="grid gap-6 lg:grid-cols-2">
            {featured.map((program) => (
              <ProgramCard key={program.slug} program={program} featured />
            ))}
          </div>
          <div className="mt-6 grid gap-6">
            {supporting.map((program) => (
              <ProgramCard key={program.slug} program={program} />
            ))}
          </div>
        </Container>
      </section>
    </>
  )
}
