(() => {
  const tabs = [...document.querySelectorAll(".model-tab")];
  const panels = {
    vit: document.getElementById("panel-vit"),
    lstm: document.getElementById("panel-lstm"),
  };

  function activate(model) {
    tabs.forEach((tab) => {
      const on = tab.dataset.model === model;
      tab.classList.toggle("is-active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
    });
    Object.entries(panels).forEach(([key, panel]) => {
      const on = key === model;
      panel.classList.toggle("is-active", on);
      panel.hidden = !on;
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activate(tab.dataset.model));
  });
})();
