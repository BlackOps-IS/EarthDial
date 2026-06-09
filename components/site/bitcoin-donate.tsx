"use client"

import { useState } from "react"
import Image from "next/image"
import { Check, Copy } from "lucide-react"
import { cn } from "@/lib/utils"

const BTC_ADDRESS = "bc1qnzv8d437a3nkea9cqp0v9x6pm403k2utftvt72349m62ehee3pvql96cat"

export function BitcoinDonate() {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(BTC_ADDRESS)
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    } catch {
      // Fallback for environments without clipboard API
      const el = document.createElement("textarea")
      el.value = BTC_ADDRESS
      el.style.position = "fixed"
      el.style.opacity = "0"
      document.body.appendChild(el)
      el.select()
      document.execCommand("copy")
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    }
  }

  return (
    <div className="flex flex-col gap-5 rounded-lg border border-border bg-card p-6 sm:p-7">
      <div className="flex items-center gap-2.5">
        {/* Bitcoin "B" mark — inline SVG keeps zero extra deps */}
        <svg
          viewBox="0 0 24 24"
          className="size-6 shrink-0 text-[#f7931a]"
          fill="currentColor"
          aria-hidden
        >
          <path d="M23.638 14.904c-1.602 6.425-8.11 10.34-14.534 8.737C2.68 22.04-1.24 15.533.362 9.107 1.962 2.682 8.47-1.232 14.894.37c6.424 1.602 10.34 8.11 8.744 14.534zm-6.272-3.61c.24-1.6-.98-2.46-2.64-3.03l.54-2.16-1.32-.33-.52 2.1c-.35-.09-.7-.17-1.05-.25l.53-2.11-1.32-.33-.54 2.16c-.29-.07-.57-.13-.85-.2v-.01l-1.82-.45-.35 1.41s.98.22.96.24c.54.13.63.49.62.77l-.63 2.53c.04.01.09.02.14.05l-.14-.03-.88 3.52c-.07.17-.24.42-.63.32.01.02-.96-.24-.96-.24l-.66 1.51 1.72.43c.32.08.63.16.94.24l-.54 2.19 1.32.33.54-2.17c.36.1.71.19 1.06.27l-.54 2.16 1.32.33.54-2.19c2.24.42 3.93.25 4.64-1.77.57-1.63-.03-2.57-1.21-3.18.86-.2 1.51-.77 1.68-1.94zm-3.01 4.22c-.4 1.62-3.14.74-4.02.52l.72-2.87c.88.22 3.7.66 3.3 2.35zm.41-4.24c-.37 1.47-2.65.73-3.39.54l.65-2.6c.74.19 3.12.53 2.74 2.06z" />
        </svg>
        <h3 className="font-serif text-lg font-medium tracking-tight">Donate with Bitcoin</h3>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        {/* QR code */}
        <div className="shrink-0">
          <Image
            src="/donate/btc-qr.png"
            alt="Bitcoin donation QR code for Black Diamond Project Corp"
            width={180}
            height={180}
            className="rounded-lg border border-border"
          />
        </div>

        {/* Address + copy */}
        <div className="flex flex-1 flex-col gap-3">
          <p className="text-sm text-muted-foreground">
            Scan the QR code or copy the address below:
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 overflow-x-auto rounded-md border border-border bg-[oklch(0.12_0.004_286)] px-3 py-2 font-mono text-[0.7rem] leading-relaxed tracking-wide text-foreground/90 sm:text-xs">
              {BTC_ADDRESS}
            </code>
            <button
              type="button"
              onClick={handleCopy}
              aria-label={copied ? "Address copied" : "Copy Bitcoin address"}
              className={cn(
                "inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border px-3 py-2 text-xs font-medium transition-colors",
                copied
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "bg-card text-muted-foreground hover:border-primary/40 hover:text-foreground",
              )}
            >
              {copied ? (
                <>
                  <Check className="size-3.5" aria-hidden />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="size-3.5" aria-hidden />
                  Copy
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <p className="border-t border-border pt-4 text-xs leading-relaxed text-muted-foreground">
        Black Diamond Project Corp is a 501(c)(3); cryptocurrency donations may be
        tax-deductible — consult your tax advisor.
      </p>
    </div>
  )
}
