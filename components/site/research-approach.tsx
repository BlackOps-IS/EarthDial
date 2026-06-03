import { researchApproach } from "@/lib/content"

export function ResearchApproachTimeline() {
  return (
    <ol className="grid gap-px overflow-hidden rounded-xl border border-border bg-border md:grid-cols-5">
      {researchApproach.map((stage) => (
        <li key={stage.step} className="flex flex-col gap-3 bg-card p-6">
          <span className="font-serif text-2xl font-medium text-primary">{stage.step}</span>
          <h3 className="text-sm font-semibold uppercase tracking-wide">{stage.title}</h3>
          <p className="text-sm leading-relaxed text-muted-foreground">{stage.body}</p>
        </li>
      ))}
    </ol>
  )
}
