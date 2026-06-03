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
    "trustworthy AI",
    "quantum resilience",
    "post-quantum cryptography",
    "public-safety technology",
    "private foundation",
    "responsible technology research",
  ],
  openGraph: {
    type: "website",
    siteName: siteConfig.organizationName,
    title: "Black Diamond Project Corp | Technology Built for Public Benefit",
    description: siteConfig.description,
    url: siteConfig.url,
  },
  twitter: {
    card: "summary_large_image",
    title: "Black Diamond Project Corp | Technology Built for Public Benefit",
    description: siteConfig.description,
  },
  robots: { index: true, follow: true },
}

export const viewport: Viewport = {
  themeColor: "#0a0a0b",
  colorScheme: "dark",
}

const organizationJsonLd = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: siteConfig.organizationName,
  url: siteConfig.url,
  logo: `${siteConfig.url}/images/irs-publication-78.png`,
  email: siteConfig.contactEmail,
  description: siteConfig.description,
  nonprofitStatus: "NonprofitType",
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
