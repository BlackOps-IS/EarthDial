import type { Metadata } from "next"
import { programDetails } from "@/lib/content"
import { ProgramDetailView } from "@/components/site/program-detail"

export const metadata: Metadata = {
  title: "AI-QEC Research Initiative | Black Diamond Project Corp",
  description:
    "AI-QEC is a Black Diamond Project Corp research initiative exploring artificial intelligence and quantum reliability.",
  alternates: { canonical: "/ai-qec" },
}

export default function AiQecPage() {
  return <ProgramDetailView detail={programDetails["ai-qec"]} />
}
