export function ImagePanel({
  b64,
  caption,
}: {
  b64: string;
  caption: string;
}) {
  return (
    <section className="panel">
      <header className="panel-header">Vista previa</header>
      <div className="p-4">
        <div className="bg-base border border-line-soft rounded-sm overflow-hidden">
          <img
            src={`data:image/png;base64,${b64}`}
            alt={caption}
            className="block w-full h-auto"
          />
        </div>
      </div>
    </section>
  );
}
