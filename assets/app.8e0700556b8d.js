"use strict";

const pages = [...document.querySelectorAll(".page")];
const navigation = [...document.querySelectorAll("[data-page]")];
const aliases = {
  "planning-options": "downloads",
  "package-installation": "downloads",
  "getting-started": "documentation",
  compiler: "documentation",
  faq: "documentation",
};

function showPage() {
  const requested = window.location.hash.slice(1);
  const resolved = aliases[requested] || requested;
  const selected = pages.find((page) => page.id === resolved) || pages[0];
  pages.forEach((page) => { page.hidden = page !== selected; });
  navigation.forEach((link) => {
    if (link.dataset.page === selected.id) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.getElementById("main").scrollTop = 0;
  document.title = selected.id === "overview"
    ? "MoFFT — Matrix-oriented fast Fourier transforms"
    : `${selected.id === "documentation" ? "Documentation" : selected.id[0].toUpperCase() + selected.id.slice(1)} — MoFFT`;
}

window.addEventListener("hashchange", showPage);
showPage();

const grid = document.getElementById("fourier-grid");
const shades = ["#36584a", "#54725a", "#7a9169", "#a6ad78", "#d1c38b"];
for (let row = 0; row < 8; row += 1) {
  for (let column = 0; column < 8; column += 1) {
    const cell = document.createElement("span");
    const phase = Math.cos((2 * Math.PI * row * column) / 8);
    cell.style.background = shades[Math.round(((phase + 1) / 2) * 4)];
    grid.appendChild(cell);
  }
}
