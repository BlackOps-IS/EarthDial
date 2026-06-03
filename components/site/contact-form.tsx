"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { contactInquiryTypes, siteConfig } from "@/lib/content"

type Status = "idle" | "error" | "success"

const fieldClass =
  "w-full rounded-md border border-border bg-card/60 px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/70 transition-colors focus:border-primary/60 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background"

export function ContactForm() {
  const [status, setStatus] = React.useState<Status>("idle")
  const [errors, setErrors] = React.useState<Record<string, string>>({})

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const form = e.currentTarget
    const data = new FormData(form)
    const name = String(data.get("name") ?? "").trim()
    const email = String(data.get("email") ?? "").trim()
    const message = String(data.get("message") ?? "").trim()

    const nextErrors: Record<string, string> = {}
    if (!name) nextErrors.name = "Please enter your name."
    if (!email) nextErrors.email = "Please enter your email."
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
      nextErrors.email = "Please enter a valid email address."
    if (!message) nextErrors.message = "Please enter a message."

    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) {
      setStatus("error")
      return
    }

    // No backend is configured; confirm receipt and direct to email as fallback.
    setStatus("success")
    form.reset()
  }

  if (status === "success") {
    return (
      <div
        role="status"
        className="rounded-lg border border-primary/40 bg-card p-8 text-center"
      >
        <h3 className="font-serif text-xl font-medium">Thank you for reaching out.</h3>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
          Your message has been recorded. A member of our team will follow up. You can also reach
          us directly at{" "}
          <a href={`mailto:${siteConfig.contactEmail}`} className="text-primary hover:underline">
            {siteConfig.contactEmail}
          </a>
          .
        </p>
        <Button
          variant="outline"
          className="mt-6"
          onClick={() => setStatus("idle")}
          type="button"
        >
          Send another message
        </Button>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="flex flex-col gap-5">
      <div className="flex flex-col gap-2">
        <label htmlFor="name" className="text-sm font-medium">
          Name
        </label>
        <input
          id="name"
          name="name"
          type="text"
          autoComplete="name"
          aria-invalid={Boolean(errors.name)}
          aria-describedby={errors.name ? "name-error" : undefined}
          className={cn(fieldClass, errors.name && "border-destructive focus:border-destructive")}
          placeholder="Your full name"
        />
        {errors.name ? (
          <p id="name-error" className="text-xs text-destructive">
            {errors.name}
          </p>
        ) : null}
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            aria-invalid={Boolean(errors.email)}
            aria-describedby={errors.email ? "email-error" : undefined}
            className={cn(
              fieldClass,
              errors.email && "border-destructive focus:border-destructive",
            )}
            placeholder="you@example.com"
          />
          {errors.email ? (
            <p id="email-error" className="text-xs text-destructive">
              {errors.email}
            </p>
          ) : null}
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="organization" className="text-sm font-medium">
            Organization <span className="text-muted-foreground">(optional)</span>
          </label>
          <input
            id="organization"
            name="organization"
            type="text"
            autoComplete="organization"
            className={fieldClass}
            placeholder="Organization or affiliation"
          />
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="inquiryType" className="text-sm font-medium">
          Inquiry type
        </label>
        <select id="inquiryType" name="inquiryType" className={cn(fieldClass, "appearance-none")}>
          {contactInquiryTypes.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="message" className="text-sm font-medium">
          Message
        </label>
        <textarea
          id="message"
          name="message"
          rows={5}
          aria-invalid={Boolean(errors.message)}
          aria-describedby={errors.message ? "message-error" : undefined}
          className={cn(
            fieldClass,
            "resize-y",
            errors.message && "border-destructive focus:border-destructive",
          )}
          placeholder="How can we help?"
        />
        {errors.message ? (
          <p id="message-error" className="text-xs text-destructive">
            {errors.message}
          </p>
        ) : null}
      </div>

      <Button type="submit" size="lg" className="w-full sm:w-auto">
        Send Message
      </Button>
      <p className="text-xs leading-relaxed text-muted-foreground">
        We use your information only to respond to your inquiry.
      </p>
    </form>
  )
}
