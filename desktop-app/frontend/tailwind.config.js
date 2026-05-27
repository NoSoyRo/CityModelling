/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    fontFamily: {
      sans: [
        "Inter",
        "ui-sans-serif",
        "system-ui",
        "-apple-system",
        "Helvetica Neue",
        "sans-serif",
      ],
      mono: [
        "JetBrains Mono",
        "SF Mono",
        "ui-monospace",
        "Menlo",
        "Consolas",
        "monospace",
      ],
    },
    fontSize: {
      "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      xs: ["0.75rem", { lineHeight: "1.1rem" }],
      sm: ["0.8125rem", { lineHeight: "1.25rem" }],
      base: ["0.9375rem", { lineHeight: "1.45rem" }],
      lg: ["1.0625rem", { lineHeight: "1.5rem" }],
      xl: ["1.25rem", { lineHeight: "1.65rem" }],
      "2xl": ["1.5rem", { lineHeight: "1.85rem" }],
      "3xl": ["2rem", { lineHeight: "2.25rem" }],
    },
    extend: {
      colors: {
        // Surfaces (deepest → lightest)
        base: "#0c0e10",
        surface: "#13161a",
        surface2: "#181c21",
        surface3: "#1f242a",
        // Borders
        line: "#262b32",
        "line-soft": "#1c2127",
        // Text
        ink: {
          DEFAULT: "#e8eaed",
          muted: "#9aa0a6",
          subtle: "#5f6368",
          inverse: "#0c0e10",
        },
        // Accents (used sparingly)
        accent: {
          DEFAULT: "#5eead4", // teal-300
          dim: "#0d9488",
        },
        warn: "#f59e0b",
        danger: "#f87171",
        ok: "#86efac",
        // Stage colour (a discrete colour per stage so the user gets
        // visual anchoring without rainbow gradients).
        stage: {
          E0: "#a3a3a3",
          E1: "#67e8f9",
          E2: "#fcd34d",
          E3: "#5eead4",
          E4: "#f0abfc",
          E5: "#86efac",
        },
      },
      boxShadow: {
        "panel": "0 1px 0 rgba(255,255,255,0.02), 0 0 0 1px rgba(0,0,0,0.6)",
      },
      letterSpacing: {
        tightish: "-0.01em",
        ui: "-0.005em",
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "5px",
        md: "6px",
        lg: "8px",
      },
    },
  },
  plugins: [],
};
