import { Card, CardContent } from "@/components/ui/card"

export function ProgramStatusPanel({
  items,
}: {
  items: { label: string; value: string }[]
}) {
  return (
    <Card className="hairline-top">
      <CardContent className="p-6 sm:p-7">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
          Status Panel
        </p>
        <dl className="mt-5 flex flex-col divide-y divide-border">
          {items.map((item) => (
            <div key={item.label} className="flex flex-col gap-1 py-3 first:pt-0 last:pb-0">
              <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {item.label}
              </dt>
              <dd className="text-sm font-medium text-foreground">{item.value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  )
}
