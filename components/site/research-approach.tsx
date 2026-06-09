import { researchApproach } from "@/lib/content"

export function ResearchApproachTimeline() {
  return (
    <ol className="grid border-y border-border md:grid-cols-5">
      {researchApproach.map((stage) => (
        <li key={stage.step} className="flex flex-col gap-3 border-b border-border py-6 md:border-b-0 md:border-r md:px-6 md:first:pl-0 md:last:border-r-0 md:last:pr-0">
          <span className="font-serif text-2xl font-medium text-primary">{stage.step}</span>
          <h3 className="text-sm font-semibold uppercase tracking-wide">{stage.title}</h3>
          <p className="text-sm leading-relaxed text-muted-foreground">{stage.body}</p>
        </li>
      ))}
    </ol>
  )
}
