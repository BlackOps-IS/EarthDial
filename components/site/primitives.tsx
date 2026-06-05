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

/** BDPROJ crest mark used in the logo lockup and favicon system. */
export function DiamondMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden
      className={cn("size-9", className)}
    >
      <path
        d="M32 4 56 20.5 46.8 58H17.2L8 20.5 32 4Z"
        fill="#07080a"
        stroke="currentColor"
        strokeWidth="3.2"
        strokeLinejoin="round"
        className="text-primary"
      />
      <path
        d="M8 20.5h48M19.5 20.5 32 4l12.5 16.5M19.5 20.5 32 58l12.5-37.5"
        stroke="currentColor"
        strokeWidth="1.45"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-primary/45"
      />
      <path
        d="M22 16.2h11.2c6.5 0 10.3 3.2 10.3 8.1 0 3.1-1.5 5.3-4.3 6.4 3.6 1 5.8 3.8 5.8 7.6 0 5.7-4.4 9.5-11.3 9.5H22V16.2Zm6.5 5.9v6.4h4.2c2.7 0 4.2-1.2 4.2-3.2 0-2.1-1.5-3.2-4.3-3.2h-4.1Zm0 11.4v8.1h5.1c3.1 0 4.8-1.5 4.8-4.1 0-2.5-1.8-4-5-4h-4.9Z"
        fill="currentColor"
        className="text-primary"
      />
    </svg>
  )
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex shrink-0 items-center gap-3", className)}>
      <span className="grid size-11 shrink-0 place-items-center rounded-sm bg-primary/10 shadow-[0_0_0_1px_rgb(242_212_122_/_0.18),0_10px_28px_rgb(0_0_0_/_0.28)]">
        <DiamondMark className="size-8" />
      </span>
      <span className="flex flex-col gap-1 leading-none">
        <span className="whitespace-nowrap font-serif text-[1.2rem] font-semibold tracking-normal text-foreground">
          BDPROJ
          <span className="ml-1 align-super font-sans text-[0.52rem] font-semibold text-primary">
            TM
          </span>
        </span>
        <span className="whitespace-nowrap text-[0.62rem] font-semibold uppercase tracking-normal text-muted-foreground">
          Black Diamond Project Corp
        </span>
      </span>
    </span>
  )
}
