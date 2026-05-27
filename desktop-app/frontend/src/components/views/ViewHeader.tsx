interface Props {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}

export function ViewHeader({ eyebrow, title, subtitle, right }: Props) {
  return (
    <div className="px-6 pt-6 pb-4 border-b border-line-soft flex items-start gap-6">
      <div className="flex-1 min-w-0">
        {eyebrow && (
          <div className="text-2xs uppercase tracking-[0.14em] text-ink-subtle mb-1">
            {eyebrow}
          </div>
        )}
        <h2 className="text-2xl tracking-tightish font-medium">{title}</h2>
        {subtitle && (
          <p className="mt-2 text-sm text-ink-muted max-w-3xl">{subtitle}</p>
        )}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}
