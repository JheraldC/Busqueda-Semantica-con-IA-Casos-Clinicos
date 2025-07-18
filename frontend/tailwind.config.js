/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{html,js,jsx,ts,tsx}",  // Incluye todos los archivos de tu src donde uses clases
    "./public/index.html",               // Si usas un index.html estático
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
