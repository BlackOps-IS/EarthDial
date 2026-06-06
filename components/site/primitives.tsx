import { cn } from "@/lib/utils"

export function Container({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={cn("mx-auto w-full max-w-6xl px-5 sm:px-8", className)}>{children}</div>
  )
}

export function Eyebrow({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return (
    <p
      className={cn(
        "text-xs font-semibold uppercase tracking-[0.22em] text-primary",
        className,
      )}
    >
      {children}
    </p>
  )
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  align = "left",
  className,
}: {
  eyebrow?: string
  title: string
  description?: string
  align?: "left" | "center"
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-4",
        align === "center" && "items-center text-center",
        className,
      )}
    >
      {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
      <h2 className="font-serif text-3xl font-medium leading-tight tracking-tight text-balance sm:text-4xl">
        {title}
      </h2>
      {description ? (
        <p
          className={cn(
            "max-w-2xl text-base leading-relaxed text-muted-foreground sm:text-lg",
            align === "center" && "mx-auto",
          )}
        >
          {description}
        </p>
      ) : null}
    </div>
  )
}

/** Faceted diamond seal used across the Black Diamond identity. */
export function DiamondMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden
      className={cn("size-9", className)}
    >
      <path
        d="M32 3 59 32 32 61 5 32 32 3Z"
        fill="currentColor"
        fillOpacity=".04"
        stroke="currentColor"
        strokeWidth="2.8"
        strokeLinejoin="round"
        className="text-primary"
      />
      <path
        d="M32 10 52 32 32 54 12 32 32 10Z"
        fill="currentColor"
        fillOpacity=".09"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
        className="text-primary/45"
      />
      <path
        d="M32 10v44M12 32h40M32 10 21 32l11 22 11-22L32 10Z"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-primary/45"
      />
      <path
        d="M32 18 43 32 32 46 21 32 32 18Z"
        fill="currentColor"
        className="text-primary"
      />
      <path
        d="M32 18v28M21 32h22"
        stroke="#07080a"
        strokeWidth="1.3"
        strokeLinecap="round"
        opacity=".7"
      />
    </svg>
  )
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex min-w-0 shrink-0 items-center gap-3", className)}>
      <span className="grid size-11 shrink-0 place-items-center rounded-sm border border-primary/25 bg-[oklch(0.12_0.006_286)] shadow-[inset_0_0_0_1px_rgb(242_212_122_/_0.06),0_10px_28px_rgb(0_0_0_/_0.28)]">
        <DiamondMark className="size-8 text-primary" />
      </span>
      <span className="flex min-w-0 flex-col items-center leading-none">
        <span className="whitespace-nowrap font-serif text-[1.08rem] font-semibold tracking-normal text-foreground sm:text-[1.18rem]">
          Black Diamond
        </span>
        <span className="my-1.5 h-px w-full bg-primary/45" aria-hidden />
        <span className="whitespace-nowrap text-[0.58rem] font-semibold uppercase tracking-normal text-primary/85 sm:text-[0.62rem]">
          Project Corp
        </span>
      </span>
    </span>
  )
}
