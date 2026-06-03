import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import { type Article } from "@/lib/content"
import { Card, CardContent } from "@/components/ui/card"

export function ArticleCard({ article }: { article: Article }) {
  return (
    <Link href={article.href} className="group block focus-visible:outline-none">
      <Card className="h-full transition-colors group-hover:border-primary/40 group-focus-visible:border-primary/60">
        <CardContent className="flex h-full flex-col gap-4 p-6">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">
              {article.category}
            </span>
            <ArrowUpRight
              className="size-4 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-primary"
              aria-hidden
            />
          </div>
          <h3 className="font-serif text-xl font-medium leading-snug tracking-tight text-balance">
            {article.title}
          </h3>
          <p className="text-sm leading-relaxed text-muted-foreground">{article.summary}</p>
          <div className="mt-auto flex items-center gap-3 border-t border-border pt-4 text-xs text-muted-foreground">
            <span className="font-medium text-foreground/80">{article.status}</span>
            <span aria-hidden>•</span>
            <span>{article.date}</span>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}
