import type { Metadata } from "next"
import { articles, techAreas } from "@/lib/content"
import { Container, SectionHeading } from "@/components/site/primitives"
import { PageHeader } from "@/components/site/page-header"
import { ArticleCard } from "@/components/site/article-card"
import { TechAreaCard } from "@/components/site/tech-area-card"
import { ResearchApproachTimeline } from "@/components/site/research-approach"

export const metadata: Metadata = {
  title: "Research & Insights",
  description:
    "Research and educational insights from Black Diamond Project Corp across secure AI, post-quantum cybersecurity, privacy-first systems including Reldun OS, and public-safety resilience technology.",
  alternates: { canonical: "/research" },
}

export default function ResearchPage() {
  return (
    <>
      <PageHeader
        eyebrow="Research"
        title="Four Connected Areas of Secure Technology"
        description="Our research spans secure and responsible AI, post-quantum cybersecurity, privacy-first systems, and public-safety resilience — connected fields where responsible design and rigorous validation matter most."
      />

      {/* Four technology areas */}
      <section className="border-b border-border py-16 sm:py-20">
        <Container>
          <div className="grid gap-5 md:grid-cols-2">
            {techAreas.map((area) => (
              <TechAreaCard key={area.slug} area={area} />
            ))}
          </div>
        </Container>
      </section>

      {/* Approach */}
      <section className="border-b border-border bg-muted/20 py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Our Approach"
            title="Research That Must Earn Trust."
            description="Every initiative moves through a disciplined path from mission need to evidence-grounded public-benefit translation."
          />
          <div className="mt-12">
            <ResearchApproachTimeline />
          </div>
        </Container>
      </section>

      {/* Featured content */}
      <section className="py-16 sm:py-20">
        <Container>
          <SectionHeading
            eyebrow="Featured"
            title="Insights & Initiative Overviews"
            description="Explore our educational post-quantum readiness guide and overviews of our flagship research initiatives."
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
