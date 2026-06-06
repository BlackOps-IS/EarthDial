import { type NextRequest, NextResponse } from "next/server"

export function proxy(request: NextRequest) {
  const incomingHost = request.headers.get("host")?.split(":")[0]

  if (incomingHost === "www.bdproj.org") {
    const canonicalUrl = request.nextUrl.clone()
    canonicalUrl.protocol = "https:"
    canonicalUrl.hostname = "bdproj.org"
    canonicalUrl.port = ""
    return NextResponse.redirect(canonicalUrl, 301)
  }

  return NextResponse.next()
}

export const config = {
  matcher: "/:path*",
}
