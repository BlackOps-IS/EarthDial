/**
 * Centralized content & configuration for Black Diamond Project Corp.
 *
 * Every public-facing claim lives here so it can be changed without editing
 * components. Do NOT add unverified claims (issued patents, awards, grants,
 * deployments, partners, EIN, street address, "501(c)(3)" or "public charity"
 * wording, "most secure / most anonymous" superlatives, etc.) to this file.
 */

export const siteConfig = {
  organizationName: "Black Diamond Project Corp",
  shortName: "Black Diamond",
  tagline:
    "Secure technology for a safer and more resilient future.",
  headline: "Secure Technology for a Safer and More Resilient Future",
  description:
    "Black Diamond Project Corp is a private foundation advancing secure AI, post-quantum cybersecurity, Reldun OS and public-safety resilience technology.",
  url: "https://www.bdproj.com",
  contactEmail: "support@bdproj.com",
  locationNeutral:
    "A private foundation advancing privacy-first, secure and resilient technology for high-trust environments.",

  // Foundation status (approved, verified facts only)
  publication78Status:
    "Black Diamond Project Corp is listed in IRS Publication 78 Data as an organization eligible to receive tax-deductible charitable contributions.",
  foundationClassification: "Private Foundation",
  deductibilityCode: "PF",
  irsVerifyUrl: "https://apps.irs.gov/app/eos/",

  // Assets
  foundationGraphic: "/images/irs-publication-78.png",
  foundationGraphicAlt:
    "Black Diamond Project Corp announcement: listed in IRS Publication 78 Data and classified as a private foundation.",
  reldunImage: "/images/reldun-os.png",
  reldunImageAlt:
    "Reldun OS emblem with the words Control Starts at the Kernel Boundary.",

  // Feature flags — keep OFF until explicitly confirmed.
  donationEnabled: false,
} as const

export const announcement = {
  message:
    "Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive tax-deductible charitable contributions.",
  ctaLabel: "View Foundation Status",
  ctaHref: "/foundation-status",
} as const

export const primaryNav = [
  { label: "Mission", href: "/mission" },
  { label: "Research", href: "/research" },
  { label: "Reldun OS", href: "/reldun-os" },
  { label: "EarthDial", href: "/earthdial" },
  { label: "Foundation Status", href: "/foundation-status" },
  { label: "Support", href: "/support" },
  { label: "Contact", href: "/contact" },
] as const

export const footerNav = {
  organization: {
    heading: "Organization",
    links: [
      { label: "Mission", href: "/mission" },
      { label: "Leadership", href: "/mission#leadership" },
      { label: "Foundation Status", href: "/foundation-status" },
      { label: "Contact", href: "/contact" },
    ],
  },
  research: {
    heading: "Research",
    links: [
      { label: "Secure AI", href: "/research" },
      { label: "Post-Quantum Security", href: "/post-quantum-security" },
      { label: "AI-QEC", href: "/ai-qec" },
      { label: "Reldun OS", href: "/reldun-os" },
      { label: "EarthDial", href: "/earthdial" },
    ],
  },
  support: {
    heading: "Support",
    links: [
      { label: "Support the Mission", href: "/support" },
      { label: "Partner With Us", href: "/contact" },
      { label: "Privacy Policy", href: "/privacy" },
      { label: "Terms of Use", href: "/terms" },
    ],
  },
} as const

export const footerLegal =
  "Black Diamond Project Corp is listed in IRS Publication 78 Data as eligible to receive tax-deductible charitable contributions. IRS deductibility code: PF — Private Foundation."

export const donationDisclosure =
  "Black Diamond Project Corp is listed in IRS Publication 78 Data as an organization eligible to receive tax-deductible charitable contributions. Its IRS deductibility code is PF, identifying it as a private foundation. Consult your tax adviser regarding your individual circumstances."

// ---- Trust strip ----------------------------------------------------------

export const trustStrip = [
  "Private Foundation",
  "Listed in IRS Publication 78 Data",
  "Secure AI Research",
  "Post-Quantum Security",
  "Public-Safety Resilience Technology",
] as const

// Compact credibility chips used in the hero.
export const heroCredibility = [
  "Private Foundation",
  "IRS Publication 78 Listed",
  "Secure & Resilient Technology Research",
] as const

// ---- Four connected technology areas --------------------------------------

export type TechArea = {
  slug: string
  name: string
  tagline: string
  summary: string
  href: string
  cta: string
}

export const techAreas: TechArea[] = [
  {
    slug: "secure-ai",
    name: "Secure and Responsible AI",
    tagline: "Privacy-first artificial intelligence",
    summary:
      "Privacy-first artificial intelligence systems designed for controlled, auditable and high-trust use, with human oversight on consequential decisions.",
    href: "/research",
    cta: "Explore Research",
  },
  {
    slug: "post-quantum",
    name: "Post-Quantum Cybersecurity",
    tagline: "Quantum-resilient protection",
    summary:
      "Research into quantum-resilient cryptography, secure communications and long-term protection of sensitive information against future threats.",
    href: "/post-quantum-security",
    cta: "Explore Post-Quantum Research",
  },
  {
    slug: "privacy-first-systems",
    name: "Privacy-First Systems / Reldun OS",
    tagline: "Control starts at the kernel boundary",
    summary:
      "Security-focused operating systems and computing architecture, including Reldun OS, built around the principle that control begins at the kernel boundary.",
    href: "/reldun-os",
    cta: "Explore Reldun OS",
  },
  {
    slug: "public-safety-resilience",
    name: "Public-Safety Resilience / EarthDial",
    tagline: "Emergency awareness and resilience",
    summary:
      "Technology initiatives such as EarthDial that support emergency awareness, preparedness, coordination and community resilience.",
    href: "/earthdial",
    cta: "Explore EarthDial",
  },
]

// ---- Mission principles & research approach -------------------------------

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
    title: "Private by Default",
    body: "Privacy and control treated as architectural principles, not afterthoughts.",
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
    body: "Architect the system around trust, resilience, privacy, and human authorization.",
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

// ---- Leadership -----------------------------------------------------------
// NOTE: Confirm final titles, biography copy, affiliations, and permission to
// publish before launch. Person schema is intentionally NOT emitted until the
// above is owner-approved.

export type Leader = {
  name: string
  role: string
  bio: string
  credentials?: string[]
}

export const leadership: Leader[] = [
  {
    name: "Simon Peter Carreras",
    role: "Founder / Lead Security Researcher",
    bio: "Simon Peter Carreras founded Black Diamond Project Corp to advance public-benefit technology at the intersection of secure AI, cybersecurity research, secure systems, and resilient public-safety technology. He guides the foundation's research direction and its commitment to disciplined, evidence-grounded innovation.",
  },
  {
    name: "Nazila Safavi, Ph.D.",
    role: "Co-Founder",
    bio: "Dr. Nazila Safavi is an engineering, computer science and information technology educator and professional whose work includes digital systems, document integrity, information technology, computer-based workflows and system reliability.",
    credentials: [
      "Ph.D. in Information Technology & Management",
      "M.S. in Telecommunications Engineering & Management, Southern Methodist University",
      "B.S. in Computer Science, Oxford Brookes University",
      "IEEE Senior Member; ACM affiliate",
    ],
  },
]

// ---- Program status styling ----------------------------------------------

export type ProgramStatus =
  | "Submitted Concept"
  | "Proposed Research"
  | "Research Initiative"

export const statusStyles: Record<
  ProgramStatus,
  { label: string; tone: "gold" | "blue" | "muted" }
> = {
  "Submitted Concept": { label: "Submitted RFI Concept", tone: "gold" },
  "Proposed Research": { label: "Proposed Research", tone: "blue" },
  "Research Initiative": { label: "Research Initiative / In Development", tone: "muted" },
}

// ---- Program / initiative detail pages ------------------------------------

export type DetailSection = { title: string; body: string; points?: string[] }

export type ProgramDetail = {
  slug: string
  name: string
  subtitle: string
  eyebrow: string
  backHref: string
  backLabel: string
  panel: { label: string; value: string }[]
  approvedMessage: string
  sections: DetailSection[]
  safetyNotice?: string
  disclosure: string
}

export const programDetails: Record<string, ProgramDetail> = {
  earthdial: {
    slug: "earthdial",
    name: "EarthDial",
    eyebrow: "Public-Safety Resilience Initiative",
    backHref: "/research",
    backLabel: "All Research",
    subtitle:
      "Emergency-awareness and resilience technology to help communities and stakeholders understand evolving hazards and response conditions.",
    panel: [
      { label: "Initiative Type", value: "Public-Safety Resilience Technology" },
      { label: "Public Status", value: "Submitted Unclassified RFI Response Concept" },
      {
        label: "Submission Context",
        value:
          "National Guard Bureau Enterprise Data and Artificial Intelligence Program RFI",
      },
      { label: "Deployment Status", value: "No deployment claim made" },
    ],
    approvedMessage:
      "EarthDial explores emergency-awareness and resilience technology intended to help communities and stakeholders better understand evolving hazards and response conditions. It is designed around federated data control, auditable decision support, source trust, and human authorization for consequential actions.",
    sections: [
      {
        title: "The Need",
        body: "Emergency awareness and community resilience depend on trusted information and decision support that hold up under disruption, uncertainty, and high-consequence conditions. Information arrives from many sources, at varying quality, and decisions often must be made quickly and accountably.",
      },
      {
        title: "The Concept",
        body: "EarthDial is a public-safety resilience initiative exploring how federated, auditable, AI-supported awareness technology can help communities and stakeholders better understand evolving hazards and response conditions. It is presented as a concept and design framework, not a deployed system.",
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
        title: "Human-Commanded Technology",
        body: "AI is positioned to support analysis and decision context. Human authorization is required for consequential actions; the concept does not propose autonomous action on critical decisions.",
      },
      {
        title: "Responsible Status Disclosure",
        body: "EarthDial is a submitted unclassified response concept. It is not deployed, selected, funded, approved, endorsed, or used by the National Guard or any government agency, and it does not provide official emergency dispatch or government-integrated services.",
      },
    ],
    safetyNotice:
      "EarthDial is a technology initiative and is not a replacement for emergency services or official emergency alerts. In an emergency, always contact your local emergency services.",
    disclosure:
      "Submitted by Black Diamond Project Corp as an unclassified response concept for the National Guard Bureau Enterprise Data and Artificial Intelligence Program RFI. No deployment, selection, funding, endorsement, or government acceptance is claimed.",
  },
  "ai-qec": {
    slug: "ai-qec",
    name: "AI-QEC",
    eyebrow: "Quantum Reliability Research Initiative",
    backHref: "/research",
    backLabel: "All Research",
    subtitle:
      "Research at the intersection of artificial intelligence and quantum reliability.",
    panel: [
      { label: "Initiative Type", value: "Quantum Reliability Research" },
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

// ---- Reldun OS ------------------------------------------------------------

export const reldunOS = {
  name: "Reldun OS",
  tagline: "Control Starts at the Kernel Boundary",
  status: "Research Initiative / In Development",
  heroCopy:
    "Reldun OS is a secure systems research initiative focused on privacy, isolation and disciplined control at the lowest levels of computing. Designed by Black Diamond Project Corp, it explores how operating-system architecture can better protect identity, activity and sensitive workflows in high-trust environments.",
  positioning:
    "Reldun OS is a privacy-first, security-focused operating system initiative designed around the principle that meaningful control begins at the kernel boundary.",
  focusAreas: [
    {
      title: "Kernel-Boundary Security Architecture",
      body: "Exploring how control and trust decisions enforced at the kernel boundary can constrain what software is permitted to do.",
    },
    {
      title: "Privacy-First System Design",
      body: "Treating privacy as an architectural property of the system rather than a configurable add-on.",
    },
    {
      title: "Capability-Restricted Access",
      body: "Researching least-privilege access principles so components receive only the capabilities they require.",
    },
    {
      title: "Controlled Exposure of Identity and Resources",
      body: "Studying how identity, activity, and system resources can be selectively and deliberately exposed.",
    },
  ],
  whyTitle: "Why Reldun OS Belongs Within the Mission",
  whyBody:
    "Secure AI and post-quantum systems are only as trustworthy as the platforms on which they operate. Reldun OS represents Black Diamond Project Corp's research into computing foundations where privacy and control are not add-ons, but architectural principles.",
  disclosure:
    "Reldun OS is an early-stage research initiative. No independent audit, deployment, adoption, benchmark result, or guarantee of anonymity or security is claimed. Capabilities described reflect research focus areas and design principles under active development.",
} as const

// ---- Research library -----------------------------------------------------

export const researchCategories = [
  {
    title: "Secure & Responsible AI",
    body: "Human oversight, adversarial evaluation, and evidence-grounded system design for high-trust use.",
  },
  {
    title: "Post-Quantum Cybersecurity",
    body: "Quantum-resilient cryptography and long-term protection of sensitive information.",
  },
  {
    title: "Privacy-First Systems",
    body: "Secure operating-system architecture and control at the kernel boundary, including Reldun OS.",
  },
  {
    title: "Public-Safety Resilience",
    body: "Resilient, auditable awareness technology for emergency preparedness and community resilience.",
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
    title: "Preparing Security for the Post-Quantum Future",
    category: "Post-Quantum Cybersecurity",
    status: "Educational Resource",
    date: "2025",
    summary:
      "Practical considerations for cryptographic inventory, quantum-risk assessment, crypto-agility, and migration planning in a changing standards environment.",
    href: "/post-quantum-security",
  },
  {
    title: "EarthDial: Public-Safety Resilience",
    category: "Public-Safety Resilience",
    status: "Submitted RFI Concept",
    date: "2025",
    summary:
      "An emergency-awareness and resilience technology initiative exploring federated, auditable decision support for evolving hazards and response conditions.",
    href: "/earthdial",
  },
  {
    title: "AI-QEC: AI and Quantum Reliability",
    category: "Post-Quantum Cybersecurity",
    status: "Proposed Research",
    date: "2025",
    summary:
      "A proposed framework exploring AI-assisted error correction, secure coordination, and digital-twin benchmarking for reliable quantum networking.",
    href: "/ai-qec",
  },
  {
    title: "Reldun OS: Control at the Kernel Boundary",
    category: "Privacy-First Systems",
    status: "Research Initiative",
    date: "2025",
    summary:
      "A privacy-first, security-focused operating system research initiative exploring secure computing foundations for high-trust environments.",
    href: "/reldun-os",
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

// ---- FAQ (answer-engine discoverability) ----------------------------------

export type Faq = { question: string; answer: string }

export const faqs: Faq[] = [
  {
    question: "What is Black Diamond Project Corp?",
    answer:
      "Black Diamond Project Corp is a private foundation and mission-driven technology research organization advancing secure artificial intelligence, post-quantum cybersecurity, privacy-first operating systems, and public-safety resilience technology.",
  },
  {
    question: "What is Reldun OS?",
    answer:
      "Reldun OS is a privacy-first, security-focused operating system research initiative from Black Diamond Project Corp, built around the principle that control starts at the kernel boundary. It is an early-stage research initiative in development.",
  },
  {
    question: "What is EarthDial?",
    answer:
      "EarthDial is a Black Diamond Project Corp public-safety resilience initiative exploring emergency-awareness technology. It is a technology initiative and not a replacement for emergency services or official emergency alerts.",
  },
  {
    question: "What is AI-QEC?",
    answer:
      "AI-QEC is a Black Diamond Project Corp research initiative exploring the intersection of artificial intelligence and quantum reliability, including AI-assisted error correction and digital-twin benchmarking. It is a proposed feasibility research initiative.",
  },
  {
    question: "Are contributions to Black Diamond Project Corp tax-deductible?",
    answer:
      "Black Diamond Project Corp is listed in IRS Publication 78 Data as an organization eligible to receive tax-deductible charitable contributions, with IRS deductibility code PF (Private Foundation). Consult your tax adviser regarding your individual circumstances.",
  },
]

// ---- Contact --------------------------------------------------------------

export const contactInquiryTypes = [
  "General Inquiry",
  "Research Collaboration",
  "Institutional Partnership",
  "Foundation / Donation Question",
  "Media Inquiry",
  "Public Information Request",
] as const
