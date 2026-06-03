import type { Metadata } from "next"
import { programDetails } from "@/lib/content"
import { ProgramDetailView } from "@/components/site/program-detail"

export const metadata: Metadata = {
  title: "EarthDial | Public-Safety Resilience Technology Initiative",
  description:
    "EarthDial is a Black Diamond Project Corp initiative exploring emergency-awareness and public-safety resilience technology.",
  alternates: { canonical: "/earthdial" },
}

export default function EarthDialPage() {
  return <ProgramDetailView detail={programDetails.earthdial} />
}
