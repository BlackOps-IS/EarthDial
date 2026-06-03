import { type Faq } from "@/lib/content"

export function FaqSection({
  items,
  withSchema = true,
}: {
  items: Faq[]
  withSchema?: boolean
}) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: items.map((f) => ({
      "@type": "Question",
      name: f.question,
      acceptedAnswer: { "@type": "Answer", text: f.answer },
    })),
  }

  return (
    <div>
      {withSchema ? (
        <script
          type="application/ld+json"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      ) : null}
      <dl className="flex flex-col divide-y divide-border">
        {items.map((item) => (
          <div key={item.question} className="py-6 first:pt-0 last:pb-0">
            <dt className="font-serif text-lg font-medium tracking-tight text-balance">
              {item.question}
            </dt>
            <dd className="mt-3 text-sm leading-relaxed text-muted-foreground">{item.answer}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
