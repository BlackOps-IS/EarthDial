/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      // New information architecture: programs → research
      { source: "/programs", destination: "/research", permanent: true },
      { source: "/programs/earthdial", destination: "/earthdial", permanent: true },
      { source: "/programs/ai-qec", destination: "/ai-qec", permanent: true },
      { source: "/research/pqc-readiness", destination: "/post-quantum-security", permanent: true },
      // Legacy / alternate slugs
      { source: "/ai", destination: "/research", permanent: true },
      { source: "/labs", destination: "/research", permanent: true },
      { source: "/security", destination: "/post-quantum-security", permanent: true },
      { source: "/post-quantum", destination: "/post-quantum-security", permanent: true },
      { source: "/quantum", destination: "/ai-qec", permanent: true },
      { source: "/reldun", destination: "/reldun-os", permanent: true },
      { source: "/os", destination: "/reldun-os", permanent: true },
      { source: "/projects", destination: "/research", permanent: true },
      { source: "/services", destination: "/research", permanent: true },
      { source: "/capabilities", destination: "/research", permanent: true },
      { source: "/methodology", destination: "/mission", permanent: true },
      { source: "/leadership", destination: "/mission#leadership", permanent: true },
      { source: "/governance", destination: "/foundation-status", permanent: true },
      { source: "/donate", destination: "/support", permanent: true },
      { source: "/partner", destination: "/contact", permanent: true },
      // Community Infrastructure Response is feature-flagged off; route stays disabled.
      { source: "/electrical", destination: "/mission", permanent: false },
    ]
  },
}

export default nextConfig
