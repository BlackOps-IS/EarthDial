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

/** Refined diamond mark used in the logo lockup and as a subtle accent. */
export function DiamondMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      className={cn("size-6", className)}
    >
      <path
        d="M6 3h12l3.5 5.2L12 21.5 2.5 8.2 6 3Z"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeLinejoin="round"
        className="text-primary"
      />
      <path
        d="M6 3l1.8 5.2h8.4L18 3M2.5 8.2h19M12 21.5l-4.2-13.3M12 21.5l4.2-13.3"
        stroke="currentColor"
        strokeWidth="0.85"
        strokeLinejoin="round"
        className="text-primary/60"
      />
    </svg>
  )
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <DiamondMark className="size-7" />
      <span className="flex flex-col leading-none">
        <span className="font-serif text-base font-semibold tracking-tight">
          Black Diamond
        </span>
        <span className="text-[0.62rem] font-medium uppercase tracking-[0.28em] text-muted-foreground">
          Project Corp
        </span>
      </span>
    </span>
  )
}
