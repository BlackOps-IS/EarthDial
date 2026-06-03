import { ShieldCheck, Layers, LineChart } from "lucide-react"
import { missionPrinciples } from "@/lib/content"
import { Card, CardContent } from "@/components/ui/card"

const icons = [ShieldCheck, Layers, LineChart]

export function MissionPrinciples() {
  return (
    <div className="grid gap-5 sm:grid-cols-3">
      {missionPrinciples.map((principle, i) => {
        const Icon = icons[i]
        return (
          <Card key={principle.title} className="hairline-top">
            <CardContent className="flex flex-col gap-4 p-7">
              <span className="inline-flex size-11 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Icon className="size-5" aria-hidden />
              </span>
              <h3 className="font-serif text-xl font-medium tracking-tight">{principle.title}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">{principle.body}</p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
