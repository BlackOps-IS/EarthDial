import type { Metadata } from "next"
import { programDetails } from "@/lib/content"
import { ProgramDetailView } from "@/components/site/program-detail"

export const metadata: Metadata = {
  title: "AI-Integrated Quantum Error Correction",
  description:
    "AI-QEC is a proposed research framework exploring AI-assisted error correction, secure coordination, and digital-twin benchmarking for reliable quantum networking. A proposed Phase I feasibility framework aligned to DOE Genesis Mission Topic 8, Focus Area D.",
}

export default function AiQecPage() {
  return <ProgramDetailView detail={programDetails["ai-qec"]} />
}
