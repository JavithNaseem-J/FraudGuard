/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        workspace: '#F8FAFC',
        surface: '#FFFFFF',
        sidebar: {
          DEFAULT: '#0F172A',
          hover: '#172554',
          border: '#1E293B',
          muted: '#94A3B8',
          text: '#CBD5E1',
        },
        primary: {
          DEFAULT: '#2563EB',
          hover: '#1D4ED8',
          soft: '#EFF6FF',
          border: '#BFDBFE',
        },
        slate: {
          main: '#0F172A',
          secondary: '#64748B',
          muted: '#94A3B8',
          border: '#E2E8F0',
        },
        status: {
          success: {
            DEFAULT: '#16A34A',
            soft: '#F0FDF4',
            border: '#BBF7D0',
          },
          warning: {
            DEFAULT: '#D97706',
            soft: '#FFFBEB',
            border: '#FEF3C7',
          },
          danger: {
            DEFAULT: '#DC2626',
            soft: '#FEF2F2',
            border: '#FECACA',
          },
        },
      },
      boxShadow: {
        subtle: '0 1px 2px 0 rgba(15, 23, 42, 0.04)',
        card: '0 1px 2px 0 rgba(15, 23, 42, 0.04)',
      },
      borderRadius: {
        card: '8px',
        input: '6px',
        btn: '6px',
        badge: '4px',
      },
      width: {
        sidebar: '240px',
      },
      spacing: {
        '18': '4.5rem',
      },
    },
  },
  plugins: [],
}
