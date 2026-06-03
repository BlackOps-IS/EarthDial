/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      { source: "/ai", destination: "/research", permanent: true },
      { source: "/labs", destination: "/programs", permanent: true },
      { source: "/security", destination: "/research", permanent: true },
      { source: "/projects", destination: "/programs", permanent: true },
      { source: "/services", destination: "/research", permanent: true },
      { source: "/capabilities", destination: "/research", permanent: true },
      { source: "/methodology", destination: "/mission", permanent: true },
      { source: "/governance", destination: "/foundation-status", permanent: true },
      // Community Infrastructure Response is feature-flagged off; route stays disabled.
      { source: "/electrical", destination: "/mission", permanent: false },
    ]
  },
}

export default nextConfig
