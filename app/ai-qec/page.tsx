import { programDetails } from "@/lib/content"
import { createPageMetadata } from "@/lib/metadata"
import { ProgramDetailView } from "@/components/site/program-detail"

export const metadata = createPageMetadata({
  title: "AI-QEC Research Initiative",
  description:
    "AI-QEC is a Black Diamond Project Corp research initiative exploring artificial intelligence and quantum reliability.",
  path: "/ai-qec",
})

export default function AiQecPage() {
  return <ProgramDetailView detail={programDetails["ai-qec"]} />
}
