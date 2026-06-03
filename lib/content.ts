/**
 * Centralized content & configuration for Black Diamond Project Corp.
 *
 * Every public-facing claim lives here so it can be changed without editing
 * components. Do NOT add unverified claims (grants, awards, deployments,
 * partners, EIN, street address, 501(c)(3) wording, etc.) to this file.
 */

export const siteConfig = {
  organizationName: "Black Diamond Project Corp",
  shortName: "Black Diamond",
  tagline: "Advancing technology for public benefit.",
  headline: "Technology Built for Public Benefit.",
  description:
    "Black Diamond Project Corp is a private foundation advancing trustworthy AI, quantum-resilient systems, and public-safety technology through responsible research.",
  url: "https://www.bdproj.com",
  contactEmail: "support@bdproj.com",
  locationNeutral:
    "A California nonprofit corporation supporting public-benefit technology initiatives in the United States.",

  // Foundation status (approved facts only)
  publication78Status:
    "Black Diamond Project Corp is listed in IRS Publication 78 as eligible to receive tax-deductible charitable contributions.",
  foundationClassification: "Private Foundation",
  deductibilityCode: "PF",

  // Assets
  foundationGraphic: "/images/irs-publication-78.png",
  foundationGraphicAlt:
    "Black Diamond Project Corp announcement: listed in IRS Publication 78 and classified as a private foundation.",

  // Feature flags — keep OFF until explicitly confirmed.
  donationEnabled: false,
  communityResponseEnabled: false,
} as const

export const announcement = {
  message:
    "Milestone: Black Diamond Project Corp is listed in IRS Publication 78 as eligible to receive tax-deductible charitable contributions.",
  ctaLabel: "View Foundation Status",
  ctaHref: "/foundation-status",
} as const

export const primaryNav = [
  { label: "Mission", href: "/mission" },
  { label: "Programs", href: "/programs" },
  { label: "Research", href: "/research" },
  { label: "Foundation Status", href: "/foundation-status" },
  { label: "About", href: "/about" },
  { label: "Contact", href: "/contact" },
] as const

export const footerNav = [
  { label: "Mission", href: "/mission" },
  { label: "Programs", href: "/programs" },
  { label: "Research", href: "/research" },
  { label: "Foundation Status", href: "/foundation-status" },
  { label: "Support", href: "/support" },
  { label: "Contact", href: "/contact" },
  { label: "Privacy", href: "/privacy" },
  { label: "Terms", href: "/terms" },
] as const

export const footerLegal =
  "Black Diamond Project Corp is a California nonprofit corporation. Listed in IRS Publication 78 as eligible to receive tax-deductible charitable contributions. IRS classification: Private Foundation."

export const donationDisclosure =
  "Black Diamond Project Corp is listed in IRS Publication 78 as eligible to receive tax-deductible charitable contributions and is classified by the IRS as a private foundation. Consult your tax adviser regarding your individual circumstances."

export type ProgramStatus =
  | "Submitted Concept"
  | "Proposed Research"
  | "Supporting Capability"

export const statusStyles: Record<
  ProgramStatus,
  { label: string; tone: "gold" | "blue" | "muted" }
> = {
  "Submitted Concept": { label: "Submitted RFI Concept", tone: "gold" },
  "Proposed Research": { label: "Proposed Research", tone: "blue" },
  "Supporting Capability": { label: "Research Foundation", tone: "muted" },
}

export type Program = {
  slug: string
  name: string
  shortName?: string
  label: string
  status: ProgramStatus
  summary: string
  footnote?: string
  href: string
  cta: string
  featured: boolean
}

export const programs: Program[] = [
  {
    slug: "earthdial",
    name: "EarthDial Guardian Mesh",
    label: "Public Safety & Mission Intelligence",
    status: "Submitted Concept",
    summary:
      "A federated, Zero Trust, AI-enabled mission intelligence concept for trusted emergency-awareness data, auditable decision support, and resilient domestic-response workflows.",
    footnote:
      "Submitted as an unclassified NGB Enterprise Data and AI Program RFI response concept.",
    href: "/programs/earthdial",
    cta: "Explore EarthDial",
    featured: true,
  },
  {
    slug: "ai-qec",
    name: "AI-Integrated Quantum Error Correction",
    shortName: "AI-QEC",
    label: "Quantum Resilience & Scientific Networking",
    status: "Proposed Research",
    summary:
      "A proposed research framework exploring AI-assisted error correction, secure coordination, and digital-twin benchmarking for reliable quantum networking environments.",
    footnote:
      "Developed as a proposed framework aligned to DOE Genesis Mission Topic 8, Focus Area D.",
    href: "/programs/ai-qec",
    cta: "Explore AI-QEC",
    featured: true,
  },
  {
    slug: "ai-security",
    name: "AI Security & Quantum Readiness",
    label: "Supporting Capability",
    status: "Supporting Capability",
    summary:
      "Adversarial evaluation, safe AI architecture, data provenance, and post-quantum readiness practices supporting responsible public-benefit technology.",
    href: "/research",
    cta: "View Research",
    featured: false,
  },
]

export const missionPrinciples = [
  {
    title: "Trustworthy by Design",
    body: "Human oversight, auditability, evidence grounding, and responsible deployment.",
  },
  {
    title: "Resilient by Purpose",
    body: "Technology designed for disruption, uncertainty, and critical operating conditions.",
  },
  {
    title: "Measurable by Method",
    body: "Research framed around validation, transparency, and reproducible results.",
  },
] as const

export const researchApproach = [
  {
    step: "01",
    title: "Mission Need",
    body: "Define the public-benefit problem and the conditions the work must serve.",
  },
  {
    step: "02",
    title: "System Design",
    body: "Architect the system around trust, resilience, and human authorization.",
  },
  {
    step: "03",
    title: "Safety & Security Review",
    body: "Evaluate risk, adversarial exposure, and failure modes before validation.",
  },
  {
    step: "04",
    title: "Controlled Validation",
    body: "Test assumptions under controlled conditions and document limitations.",
  },
  {
    step: "05",
    title: "Public-Benefit Translation",
    body: "Describe only what the work can support, grounded in evidence.",
  },
] as const

export const heroCredibility = [
  "IRS Publication 78 Listed",
  "Private Foundation",
  "Responsible Technology Research",
] as const

// ---- Program detail pages -------------------------------------------------

export type DetailSection = { title: string; body: string; points?: string[] }

export type ProgramDetail = {
  slug: string
  name: string
  subtitle: string
  panel: { label: string; value: string }[]
  approvedMessage: string
  sections: DetailSection[]
  disclosure: string
}

export const programDetails: Record<string, ProgramDetail> = {
  earthdial: {
    slug: "earthdial",
    name: "EarthDial Guardian Mesh",
    subtitle:
      "Trusted AI decision-support concepts for emergency awareness and resilient operations.",
    panel: [
      { label: "Initiative Type", value: "Public Safety & Mission Intelligence Research" },
      { label: "Public Status", value: "Submitted Unclassified RFI Response Concept" },
      {
        label: "Submission Context",
        value:
          "National Guard Bureau Enterprise Data and Artificial Intelligence Program RFI",
      },
      { label: "Deployment Status", value: "No deployment claim made" },
    ],
    approvedMessage:
      "EarthDial Guardian Mesh is designed around federated data control, auditable AI decision support, source trust, resilient operations, and human authorization for consequential actions.",
    sections: [
      {
        title: "The Need",
        body: "Emergency awareness and domestic-response coordination depend on trusted data and decision support that hold up under disruption, uncertainty, and high-consequence conditions. Information arrives from many sources, at varying quality, and decisions often must be made quickly and accountably.",
      },
      {
        title: "The Concept",
        body: "EarthDial Guardian Mesh is a federated, Zero Trust, AI-enabled mission intelligence concept designed to support trusted operational data, emergency awareness, auditable decision support, and resilient domestic-response workflows. It is presented as a concept and design framework, not a deployed system.",
      },
      {
        title: "Design Principles",
        body: "The concept is organized around a small set of principles intended to keep it trustworthy and accountable in practice.",
        points: [
          "Federated data control rather than centralized aggregation",
          "Source trust and verifiable data provenance",
          "Auditability of every decision-support pathway",
          "Resilient operation under degraded or disconnected conditions",
        ],
      },
      {
        title: "Human-Commanded AI",
        body: "AI is positioned to support analysis and decision context. Human authorization is required for consequential actions; the concept does not propose autonomous action on critical decisions.",
      },
      {
        title: "Trust, Auditability, and Zero Trust Architecture",
        body: "A Zero Trust posture, with verifiable sources and auditable decision pathways, is intended to make the system's behavior reviewable and accountable rather than opaque.",
      },
      {
        title: "Responsible Status Disclosure",
        body: "EarthDial Guardian Mesh is a submitted unclassified response concept. It is not deployed, selected, funded, approved, endorsed, or used by the National Guard or any government agency.",
      },
    ],
    disclosure:
      "Submitted by Black Diamond Project Corp as an unclassified response concept for the National Guard Bureau Enterprise Data and Artificial Intelligence Program RFI. No deployment, selection, funding, endorsement, or government acceptance is claimed.",
  },
  "ai-qec": {
    slug: "ai-qec",
    name: "AI-Integrated Quantum Error Correction",
    subtitle: "Exploring AI-assisted reliability for scientific quantum networking.",
    panel: [
      { label: "Initiative Type", value: "Quantum Resilience Research" },
      { label: "Public Status", value: "Proposed Feasibility Research Initiative" },
      { label: "Alignment Context", value: "DOE Genesis Mission Topic 8, Focus Area D" },
      { label: "Funding Status", value: "No award or funding claim made" },
    ],
    approvedMessage:
      "AI-QEC is a proposed research framework exploring predictive error-syndrome inference, secure coordination, digital-twin benchmarking, and multi-node orchestration for more reliable quantum computing and scientific networking environments.",
    sections: [
      {
        title: "Scientific Challenge",
        body: "Quantum computing and scientific networking environments are highly sensitive to error and disruption. Improving reliability requires better error correction, secure coordination across nodes, and rigorous benchmarking before any performance claim can be made.",
      },
      {
        title: "Proposed Research Framework",
        body: "AI-QEC is a proposed framework exploring AI-assisted approaches to reliability across quantum networking environments.",
        points: [
          "Predictive error-syndrome inference",
          "Secure classical coordination",
          "Multi-session orchestration",
          "Digital-twin benchmarking",
        ],
      },
      {
        title: "Digital Twin and Benchmarking Approach",
        body: "A digital-twin approach is proposed to benchmark behavior under controlled conditions before drawing any conclusions about reliability or performance, keeping evaluation grounded and reproducible.",
      },
      {
        title: "Evaluation Metrics",
        body: "Evaluation would be framed around measurable indicators rather than claimed outcomes.",
        points: [
          "Logical error rate",
          "State fidelity",
          "Latency and throughput",
          "Resource efficiency",
        ],
      },
      {
        title: "Responsible Status Disclosure",
        body: "AI-QEC is a proposed feasibility research initiative. It has not received DOE funding, DOE approval, selection, award, university sponsorship, or a formal partnership, and no validated breakthroughs or operational quantum deployments are claimed.",
      },
    ],
    disclosure:
      "Developed as a proposed Phase I feasibility framework aligned to DOE Genesis Mission Topic 8, Focus Area D — AI for Quantum Computing and Networking. No funding, award, selection, sponsorship, or partnership is claimed.",
  },
}

// ---- Research library -----------------------------------------------------

export const researchCategories = [
  {
    title: "Trustworthy AI",
    body: "Human oversight, adversarial evaluation, and evidence-grounded system design.",
  },
  {
    title: "Quantum Resilience",
    body: "AI-assisted error correction and post-quantum readiness for critical systems.",
  },
  {
    title: "Cybersecurity Assurance",
    body: "Data provenance, secure architecture, and disciplined validation practices.",
  },
  {
    title: "Public Safety Technology",
    body: "Resilient, auditable decision support for emergency awareness and response.",
  },
] as const

export type Article = {
  title: string
  category: string
  status: string
  date: string
  summary: string
  href: string
}

export const articles: Article[] = [
  {
    title: "Post-Quantum Cryptography Readiness Guide",
    category: "Quantum Resilience",
    status: "Educational Resource",
    date: "2025",
    summary:
      "Practical considerations for cryptographic inventory, quantum-risk assessment, crypto-agility, and migration planning in a changing standards environment.",
    href: "/research/pqc-readiness",
  },
  {
    title: "EarthDial Guardian Mesh Overview",
    category: "Public Safety Technology",
    status: "Submitted RFI Concept",
    date: "2025",
    summary:
      "A federated, Zero Trust, AI-enabled mission intelligence concept for trusted emergency-awareness data and auditable decision support.",
    href: "/programs/earthdial",
  },
  {
    title: "AI-QEC Research Overview",
    category: "Quantum Resilience",
    status: "Proposed Research",
    date: "2025",
    summary:
      "A proposed framework exploring AI-assisted error correction, secure coordination, and digital-twin benchmarking for reliable quantum networking.",
    href: "/programs/ai-qec",
  },
]

export const pqcSections: DetailSection[] = [
  {
    title: "Cryptographic Inventory",
    body: "Understanding what cryptographic systems exist across an organization is the essential first step. This includes identifying where public-key cryptography is used, what algorithms are employed, key sizes, and dependencies on cryptographic libraries and protocols.",
    points: [
      "Map all systems using RSA, ECDSA, ECDH, and Diffie-Hellman",
      "Document key sizes and certificate chains",
      "Identify cryptographic library and protocol dependencies",
      "Catalog TLS/SSL implementations and versions",
    ],
  },
  {
    title: "Quantum-Risk Assessment",
    body: "Not all cryptographic applications face the same timeline or severity of quantum risk. Data with long confidentiality requirements faces harvest-now-decrypt-later threats, while authentication systems may have a different risk profile.",
    points: [
      "Identify data with long-term confidentiality needs",
      "Assess exposure to harvest-now-decrypt-later threats",
      "Weigh authentication versus encryption priorities",
      "Consider regulatory and compliance timelines",
    ],
  },
  {
    title: "Crypto-Agility",
    body: "Building systems that can transition between cryptographic algorithms without major architectural changes is increasingly important. Crypto-agility reduces migration risk and enables faster response to emerging threats or standards changes.",
    points: [
      "Design for algorithm abstraction layers",
      "Plan for hybrid classical and post-quantum modes",
      "Enable certificate and key rotation mechanisms",
      "Test migration paths before they are needed",
    ],
  },
  {
    title: "Migration Planning",
    body: "NIST has standardized post-quantum algorithms including ML-KEM (formerly CRYSTALS-Kyber) for key encapsulation and ML-DSA (formerly CRYSTALS-Dilithium) for digital signatures. Planning migration requires understanding these standards and their implementation requirements.",
    points: [
      "Monitor NIST post-quantum standardization progress",
      "Evaluate ML-KEM and ML-DSA implementations",
      "Phase migration starting with highest-risk systems",
      "Coordinate with vendors and supply-chain partners",
    ],
  },
]

export const contactInquiryTypes = [
  "General Inquiry",
  "Research Collaboration",
  "Public Safety Technology Inquiry",
  "Institutional / Government Inquiry",
  "Support or Donation Question",
  "Media Inquiry",
] as const
