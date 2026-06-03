import type { Metadata, Viewport } from "next"
import { Inter, Fraunces } from "next/font/google"
import "./globals.css"
import { siteConfig } from "@/lib/content"
import { AnnouncementBar } from "@/components/site/announcement-bar"
import { SiteHeader } from "@/components/site/site-header"
import { SiteFooter } from "@/components/site/site-footer"

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
    default: "Black Diamond Project Corp | Technology Built for Public Benefit",
    template: "%s | Black Diamond Project Corp",
  },
  description: siteConfig.description,
  keywords: [
    "secure AI",
    "responsible artificial intelligence",
    "post-quantum cybersecurity",
    "post-quantum cryptography",
    "Reldun OS",
    "privacy-first operating system",
    "public-safety resilience technology",
    "EarthDial",
    "AI-QEC",
    "private foundation",
  ],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: siteConfig.organizationName,
    title: "Black Diamond Project Corp | Secure Technology for a Safer Future",
    description: siteConfig.description,
    url: siteConfig.url,
    images: [
      {
        url: "/images/og-default.png",
        width: 1200,
        height: 630,
        alt: "Black Diamond Project Corp",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Black Diamond Project Corp | Secure Technology for a Safer Future",
    description: siteConfig.description,
    images: ["/images/og-default.png"],
  },
  robots: { index: true, follow: true },
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
  sameAs: [siteConfig.earthDialUrl],
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
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[100] focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
        >
          Skip to content
        </a>
        <AnnouncementBar />
        <SiteHeader />
        <main id="main">{children}</main>
        <SiteFooter />
      </body>
    </html>
  )
}
