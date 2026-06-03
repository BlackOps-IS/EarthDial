import type { Metadata } from "next"
import { articles, researchCategories } from "@/lib/content"
import { Container, SectionHeading } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { ArticleCard } from "@/components/site/article-card"

export const metadata: Metadata = {
  title: "Research & Insights",
  description:
    "Research and educational insights from Black Diamond Project Corp across trustworthy AI, quantum resilience, cybersecurity assurance, and public-safety technology.",
}

export default function ResearchPage() {
  return (
    <>
      <PageHeader
        eyebrow="Research"
        title="Research & Insights"
        description="Educational resources and initiative overviews across the fields where responsible design and rigorous validation matter most."
      />

      {/* Categories */}
      <section className="border-b border-border py-14">
        <Container>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {researchCategories.map((category) => (
              <div
                key={category.title}
                className="flex flex-col gap-2 rounded-lg border border-border bg-card p-5"
              >
                <h2 className="font-serif text-lg font-medium tracking-tight">{category.title}</h2>
                <p className="text-sm leading-relaxed text-muted-foreground">{category.body}</p>
              </div>
            ))}
          </div>
        </Container>
      </section>

      {/* Featured content */}
      <section className="py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Featured"
            title="Insights & Initiative Overviews"
            description="Explore our educational PQC readiness guide and overviews of our flagship research initiatives."
          />
          <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {articles.map((article) => (
              <ArticleCard key={article.href} article={article} />
            ))}
          </div>
        </Container>
      </section>
    </>
  )
}
