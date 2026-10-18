"use strict";
const controls = document.querySelector(".gallery-controls");
const buttons = Array.from(document.querySelectorAll("[data-gallery]"));
const panels = Array.from(document.querySelectorAll("[data-panel]"));
function selectGallery(name) {
  buttons.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.gallery === name)));
  panels.forEach(panel => { panel.hidden = panel.dataset.panel !== name; });
}
if (controls && buttons.length && panels.length) {
  controls.hidden = false;
  buttons.forEach(button => button.addEventListener("click", () => selectGallery(button.dataset.gallery)));
  selectGallery("navsim");
}
