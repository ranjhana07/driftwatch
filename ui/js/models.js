export function formatTime(iso) {
  if (!iso) {
    return "never";
  }
  const date = new Date(iso);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function renderModels(container, models, selected, onSelect) {
  container.replaceChildren();
  if (models.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No models yet. Send events with the SDK or run the demo producer.";
    container.append(empty);
    return;
  }
  models.forEach((model) => {
    const level = model.status.level;
    const card = document.createElement("div");
    card.className = `card level-${level}` + (model.name === selected ? " selected" : "");
    card.tabIndex = 0;

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = model.name;

    const badge = document.createElement("span");
    badge.className = `badge ${level}`;
    badge.textContent = level === "none" ? "healthy" : level;

    const head = document.createElement("div");
    head.className = "row";
    head.append(name, badge);

    const events = document.createElement("div");
    events.className = "row";
    events.textContent = `${model.events.toLocaleString()} events`;

    const flagged = document.createElement("div");
    flagged.className = "row";
    flagged.textContent = `${model.status.flagged} flagged in last window`;

    const last = document.createElement("div");
    last.className = "row";
    last.textContent = `last event ${formatTime(model.last_event)}`;

    card.append(head, events, flagged, last);
    card.addEventListener("click", () => onSelect(model.name));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        onSelect(model.name);
      }
    });
    container.append(card);
  });
}
