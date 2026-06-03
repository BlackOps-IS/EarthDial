import type { Metadata } from "next"
import { programDetails } from "@/lib/content"
import { ProgramDetailView } from "@/components/site/program-detail"

export const metadata: Metadata = {
  title: "EarthDial Guardian Mesh",
  description:
    "EarthDial Guardian Mesh is a federated, Zero Trust, AI-enabled mission intelligence concept for trusted emergency-awareness data, auditable decision support, and resilient domestic-response workflows. Submitted as an unclassified NGB RFI response concept.",
}

export default function EarthDialPage() {
  return <ProgramDetailView detail={programDetails.earthdial} />
}
