import type { Metadata, Viewport } from "next"
import { Inter, Fraunces } from "next/font/google"
import "./globals.css"
import { leadership, siteConfig } from "@/lib/content"
import { AnnouncementBar } from "@/components/site/announcement-bar"
import { SiteHeader } from "@/components/site/site-header"
import { SiteFooter } from "@/components/site/site-footer"
import { RouteBreadcrumbs } from "@/components/site/route-breadcrumbs"

const siteTitle = "Black Diamond Project Corp | Secure Technology for a Safer Future"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["opsz"],
})

export const metadata: Metadata = {
  metadataBase: new URL(siteConfig.url),
  title: {
    default: siteTitle,
    template: "%s | Black Diamond Project Corp",
  },
  applicationName: siteConfig.organizationName,
  description: siteConfig.description,
  manifest: "/site.webmanifest",
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: siteConfig.organizationName,
    title: siteTitle,
    description: siteConfig.description,
    url: siteConfig.url,
    images: [
      {
        url: `${siteConfig.url}/images/og-default.png`,
        width: 1200,
        height: 630,
        alt: "Black Diamond Project Corp",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteTitle,
    description: siteConfig.description,
    images: [`${siteConfig.url}/images/og-default.png`],
  },
  robots: { index: true, follow: true },
  icons: {
    icon: [
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon.ico", sizes: "any" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
  },
}

export const viewport: Viewport = {
  themeColor: "#0a0a0b",
  colorScheme: "dark",
}

const organizationJsonLd = {
  "@context": "https://schema.org",
  "@type": ["Organization", "NGO"],
  name: siteConfig.organizationName,
  url: siteConfig.url,
  logo: `${siteConfig.url}/images/og-default.png`,
  image: `${siteConfig.url}/images/og-default.png`,
  email: siteConfig.contactEmail,
  description: siteConfig.description,
  slogan: siteConfig.tagline,
  nonprofitStatus: "NonprofitType",
  sameAs: [
    siteConfig.earthDialUrl,
    siteConfig.linkedInUrl,
    "https://www.linkedin.com/in/simoncarreras/",
    "https://www.linkedin.com/in/dr-nazila-safavi-267ba65/",
    "https://foothill.ieee-bv.org/2026/02/foothill-sections-new-cs-and-local-blockchain-group-chair-prof-nazila-safavi/",
    "https://events.vtools.ieee.org/m/514916",
  ],
  knowsAbout: [
    "Secure and Responsible Artificial Intelligence",
    "Post-Quantum Cybersecurity",
    "Privacy-First Operating Systems",
    "Reldun OS",
    "Public-Safety Resilience Technology",
    "EarthDial",
    "AI-QEC",
  ],
}

const websiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: siteConfig.organizationName,
  url: siteConfig.url,
  description: siteConfig.description,
  publisher: { "@type": "Organization", name: siteConfig.organizationName },
}

const leadershipJsonLd = leadership.map((person) => ({
  "@context": "https://schema.org",
  "@type": "Person",
  name: person.name,
  jobTitle: person.role,
  description: person.bio,
  worksFor: {
    "@type": "Organization",
    name: siteConfig.organizationName,
    url: siteConfig.url,
  },
  affiliation: [
    {
      "@type": "Organization",
      name: siteConfig.organizationName,
      url: siteConfig.url,
    },
    {
      "@type": "Organization",
      name: "IEEE",
      url: "https://www.ieee.org",
    },
  ],
  sameAs: person.links?.filter((link) => link.href.includes("linkedin.com")).map((link) => link.href) ?? [],
  ...(person.credentials
    ? {
        hasCredential: person.credentials.map((credential) => ({
          "@type": "EducationalOccupationalCredential",
          name: credential,
        })),
      }
    : {}),
}))

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable} bg-background`}>
      <body className="min-h-dvh antialiased">
        <script
          type="application/ld+json"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
        />
        <script
          type="application/ld+json"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
        {leadershipJsonLd.map((person) => (
          <script
            key={person.name}
            type="application/ld+json"
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: JSON.stringify(person) }}
          />
        ))}
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[100] focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
        >
          Skip to content
        </a>
        <AnnouncementBar />
        <SiteHeader />
        <main id="main">
          <RouteBreadcrumbs />
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  )
}
