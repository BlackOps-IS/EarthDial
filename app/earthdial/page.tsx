import { programDetails } from "@/lib/content"
import { createPageMetadata } from "@/lib/metadata"
import { ProgramDetailView } from "@/components/site/program-detail"

export const metadata = createPageMetadata({
  title: "EarthDial | Public-Safety Resilience Technology Initiative",
  description:
    "EarthDial is a Black Diamond Project Corp initiative exploring emergency-awareness and public-safety resilience technology.",
  path: "/earthdial",
})

export default function EarthDialPage() {
  return <ProgramDetailView detail={programDetails.earthdial} />
}
