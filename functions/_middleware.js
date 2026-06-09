export async function onRequest(context) {
  const url = new URL(context.request.url)

  if (url.hostname === "www.bdproj.org") {
    url.hostname = "bdproj.org"
    return Response.redirect(url.toString(), 301)
  }

  if (url.pathname.indexOf("/", 1) !== -1 && url.pathname.endsWith(".__PAGE__.txt")) {
    url.pathname = url.pathname.replace(".__PAGE__.txt", "/__PAGE__.txt")
    return context.env.ASSETS.fetch(new Request(url, context.request))
  }

  return context.next()
}
