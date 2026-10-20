"use strict";

const repository = "https://github.com/ruge-zhang/MoFFT";
const grid = document.getElementById("fourier-grid");
for (let row = 0; row < 8; row += 1) {
  for (let column = 0; column < 8; column += 1) {
    const cell = document.createElement("span");
    const phase = Math.cos((2 * Math.PI * row * column) / 8);
    const shades = ["#36584a", "#54725a", "#7a9169", "#a6ad78", "#d1c38b"];
    cell.style.background = shades[Math.round(((phase + 1) / 2) * 4)];
    if (row === 3) cell.classList.add("highlight");
    grid.appendChild(cell);
  }
}

const buildExample = `git clone https://github.com/ruge-zhang/MoFFT.git
cd MoFFT

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure`;

function cExample(precision) {
  return `#include <mofft.h>

int main(void) {
  mofft_complex_${precision} input[1024] = {0};
  mofft_complex_${precision} output[1024];
  mofft_plan_${precision} *plan = NULL;

  mofft_status status = mofft_plan_create_${precision}(
      &plan, 1024, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  if (status != MOFFT_SUCCESS) return status;

  status = mofft_execute_${precision}(plan, input, output);
  mofft_plan_destroy_${precision}(plan);
  return status;
}`;
}

const examples = { build: buildExample, fp32: cExample("f32"), fp64: cExample("f64") };
const codeTabs = [...document.querySelectorAll(".code-tab")];
const code = document.getElementById("code-example");
const codeContent = document.getElementById("code-content");
const copyButton = document.getElementById("copy-code");
const copyStatus = document.getElementById("copy-status");
let selectedCode = "build";
let copyTimer;

function selectCode(tab) {
  selectedCode = tab.dataset.code;
  for (const button of codeTabs) {
    const active = button === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
  code.textContent = examples[selectedCode];
  codeContent.setAttribute("aria-labelledby", tab.id);
  document.getElementById("code-language").textContent = selectedCode === "build" ? "TERMINAL" : "C · MOFFT API";
  copyButton.setAttribute("aria-label", `Copy ${selectedCode === "build" ? "build commands" : `${selectedCode.toUpperCase()} C example`}`);
  copyButton.querySelector("span").textContent = "Copy";
  copyStatus.textContent = "";
  clearTimeout(copyTimer);
}
codeTabs.forEach((tab) => tab.addEventListener("click", () => selectCode(tab)));
selectCode(codeTabs[0]);

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(examples[selectedCode]);
    copyButton.querySelector("span").textContent = "Copied";
    copyStatus.textContent = "Code copied to clipboard.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.querySelector("span").textContent = "Select & copy";
    copyStatus.textContent = "Code selected. Use your browser's copy command.";
  }
  clearTimeout(copyTimer);
  copyTimer = setTimeout(() => { copyButton.querySelector("span").textContent = "Copy"; }, 2500);
});

const stages = [
  { type: "TARGET PROFILE", title: "Make the target part of the model.", text: "Instruction and memory measurements inform a machine profile. The compiler uses that profile to estimate the cost of candidate computations.", link: `${repository}/tree/main/profiles`, label: "View target profiles" },
  { type: "EQUIVALENT EXPRESSIONS", title: "Explore the structure of the butterfly.", text: "Equivalent expressions offer different computation and data-movement costs. A typed dependency graph makes those choices available to the compiler.", link: `${repository}/tree/main/compiler/mofft_compiler`, label: "Explore the compiler" },
  { type: "SCHEDULE & LOWER", title: "Turn dependencies into executable kernels.", text: "The selected graph is scheduled and lowered to Arm SME/SME2 ACLE intrinsics. CMake generates the kernel sources and integrates them into the library.", link: `${repository}/tree/main/compiler/mofft_compiler/backends`, label: "View code generation" },
  { type: "PLAN WISDOM", title: "Compose kernels into a complete transform.", text: "The runtime combines radix kernels into FFT plans. Search-based wisdom records measured plan choices for the matching target and generated kernels.", link: `${repository}/blob/main/tools/plan_search.cpp`, label: "Explore plan search" }
];
const stageTabs = [...document.querySelectorAll(".pipeline-step")];
function selectStage(tab) {
  stageTabs.forEach((button) => {
    const active = button === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  const stage = stages[Number(tab.dataset.stage)];
  document.getElementById("pipeline-type").textContent = stage.type;
  document.getElementById("pipeline-title").textContent = stage.title;
  document.getElementById("pipeline-text").textContent = stage.text;
  const link = document.getElementById("pipeline-link");
  link.href = stage.link;
  link.textContent = `${stage.label} ↗`;
  document.getElementById("pipeline-detail").setAttribute("aria-labelledby", tab.id);
}
stageTabs.forEach((tab) => tab.addEventListener("click", () => selectStage(tab)));

function addTabKeys(tabs) {
  tabs.forEach((tab, index) => tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      tabs[next].focus();
      tabs[next].click();
    }
  }));
}
addTabKeys(codeTabs);
addTabKeys(stageTabs);

const menuToggle = document.querySelector(".menu-toggle");
const nav = document.getElementById("site-nav");
function closeMenu() {
  menuToggle.setAttribute("aria-expanded", "false");
  menuToggle.setAttribute("aria-label", "Open navigation");
  nav.classList.remove("is-open");
}
menuToggle.addEventListener("click", () => {
  const expanded = menuToggle.getAttribute("aria-expanded") !== "true";
  menuToggle.setAttribute("aria-expanded", String(expanded));
  menuToggle.setAttribute("aria-label", expanded ? "Close navigation" : "Open navigation");
  nav.classList.toggle("is-open", expanded);
});
nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(); });

const installationDetails = document.getElementById("package-installation");
document.querySelectorAll('a[href="#package-installation"]').forEach((link) => {
  link.addEventListener("click", () => { installationDetails.open = true; });
});
if (window.location.hash === "#package-installation") installationDetails.open = true;
