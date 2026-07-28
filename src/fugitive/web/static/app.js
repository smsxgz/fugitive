"use strict";

const API_ROOT = "/api";
const PILE_RANGES = ["4–14", "15–28", "29–41"];
const MAX_SEED = 18446744073709551615n;

const PHASE_LABELS = {
  fugitive_opening: "Fugitive 开局出牌",
  fugitive_draw: "Fugitive 抽牌",
  fugitive_action: "Fugitive 出牌",
  marshal_draw: "Marshal 抽牌",
  marshal_guess: "Marshal 猜测",
  manhunt: "最终追捕",
  terminal: "对局结束",
};

const MODE_LABELS = {
  human_fugitive: "扮演 Fugitive",
  human_marshal: "扮演 Marshal",
  spectate: "观战",
};

const ROLE_LABELS = {
  fugitive: "Fugitive",
  marshal: "Marshal",
  draw: "平局",
  spectator: "观战",
};

const SPECTATOR_VIEW_LABELS = {
  omniscient: "全知视角",
  fugitive: "Fugitive 视角",
  marshal: "Marshal 视角",
  public: "公开视角",
};

const REASON_LABELS = {
  escape_no_manhunt: "成功抵达 42，无最终追捕",
  manhunt_failed: "Marshal 在最终追捕中猜错",
  manhunt_success: "Marshal 完成最终追捕",
  marshal_uncovered_route: "Marshal 揭开全部藏身处",
  max_rounds: "达到研究版最大回合数",
};

const elements = {
  connectionStatus: document.querySelector("#connectionStatus"),
  connectionLabel: document.querySelector("#connectionStatus .connection-label"),
  setupPanel: document.querySelector("#setupPanel"),
  setupForm: document.querySelector("#setupForm"),
  fugitiveAgent: document.querySelector("#fugitiveAgent"),
  marshalAgent: document.querySelector("#marshalAgent"),
  fugitiveHuman: document.querySelector("#fugitiveHuman"),
  marshalHuman: document.querySelector("#marshalHuman"),
  gameSeed: document.querySelector("#gameSeed"),
  startGameButton: document.querySelector("#startGameButton"),
  gameShell: document.querySelector("#gameShell"),
  roundBadge: document.querySelector("#roundBadge"),
  phaseLabel: document.querySelector("#phaseLabel"),
  turnLabel: document.querySelector("#turnLabel"),
  sessionIdLabel: document.querySelector("#sessionIdLabel"),
  resetButton: document.querySelector("#resetButton"),
  newGameButton: document.querySelector("#newGameButton"),
  outcomeBanner: document.querySelector("#outcomeBanner"),
  outcomeTitle: document.querySelector("#outcomeTitle"),
  outcomeReason: document.querySelector("#outcomeReason"),
  pileList: document.querySelector("#pileList"),
  routeTrack: document.querySelector("#routeTrack"),
  routeSummary: document.querySelector("#routeSummary"),
  handCards: document.querySelector("#handCards"),
  handSummary: document.querySelector("#handSummary"),
  actionDock: document.querySelector("#actionDock"),
  actionStatus: document.querySelector("#actionStatus"),
  drawAction: document.querySelector("#drawAction"),
  fugitiveAction: document.querySelector("#fugitiveAction"),
  marshalAction: document.querySelector("#marshalAction"),
  waitingAction: document.querySelector("#waitingAction"),
  waitingText: document.querySelector("#waitingText"),
  selectedHideout: document.querySelector("#selectedHideout"),
  selectedSprint: document.querySelector("#selectedSprint"),
  passButton: document.querySelector("#passButton"),
  playHideoutButton: document.querySelector("#playHideoutButton"),
  guessGrid: document.querySelector("#guessGrid"),
  guessCount: document.querySelector("#guessCount"),
  clearGuessButton: document.querySelector("#clearGuessButton"),
  submitGuessButton: document.querySelector("#submitGuessButton"),
  spectatorControls: document.querySelector("#spectatorControls"),
  spectatorViewSelect: document.querySelector("#spectatorViewSelect"),
  autoState: document.querySelector("#autoState"),
  stepButton: document.querySelector("#stepButton"),
  autoButton: document.querySelector("#autoButton"),
  continueButton: document.querySelector("#continueButton"),
  terminateButton: document.querySelector("#terminateButton"),
  speedSelect: document.querySelector("#speedSelect"),
  modeValue: document.querySelector("#modeValue"),
  fugitiveValue: document.querySelector("#fugitiveValue"),
  marshalValue: document.querySelector("#marshalValue"),
  viewerValue: document.querySelector("#viewerValue"),
  seedValue: document.querySelector("#seedValue"),
  copySeedButton: document.querySelector("#copySeedButton"),
  exportTraceButton: document.querySelector("#exportTraceButton"),
  logCount: document.querySelector("#logCount"),
  gameLog: document.querySelector("#gameLog"),
  toast: document.querySelector("#toast"),
};

const app = {
  game: null,
  gameId: null,
  agents: { fugitive: [], marshal: [] },
  busy: false,
  cardMode: "hideout",
  selectedHideout: null,
  selectedSprint: new Set(),
  fugitiveDraft: null,
  fugitiveDraftToken: 0,
  selectedGuesses: new Set(),
  autoRunning: false,
  autoTimer: null,
  toastTimer: null,
};

class APIError extends Error {
  constructor(message, code, status) {
    super(message);
    this.name = "APIError";
    this.code = code;
    this.status = status;
  }
}

function getMode() {
  return new FormData(elements.setupForm).get("mode") || "human_fugitive";
}

function setConnection(kind, label) {
  elements.connectionStatus.classList.toggle("is-online", kind === "online");
  elements.connectionStatus.classList.toggle("is-error", kind === "error");
  elements.connectionLabel.textContent = label;
}

function setBusy(busy) {
  app.busy = busy;
  document.body.classList.toggle("is-busy", busy);
  renderInteractiveState();
}

function showToast(message) {
  window.clearTimeout(app.toastTimer);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  app.toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 5200);
}

async function requestJSON(path, options = {}) {
  const init = {
    method: options.method || "GET",
    headers: { Accept: "application/json" },
  };
  if (Object.prototype.hasOwnProperty.call(options, "body")) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }

  let response;
  try {
    response = await fetch(`${API_ROOT}${path}`, init);
  } catch (error) {
    setConnection("error", "连接失败");
    throw new APIError("无法连接本地 Fugitive 服务", "network_error", 0);
  }

  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      payload = await response.json();
    } catch (_error) {
      throw new APIError("服务器返回了无效 JSON", "invalid_response", response.status);
    }
  }

  if (!response.ok) {
    const detail = payload && payload.error;
    const message =
      (detail && typeof detail.message === "string" && detail.message) ||
      `请求失败 (${response.status})`;
    const code = (detail && detail.code) || "request_failed";
    throw new APIError(message, code, response.status);
  }

  setConnection("online", "已连接");
  return payload;
}

function normalizeAgentList(payload, role) {
  const source = payload && Array.isArray(payload[role]) ? payload[role] : [];
  return source
    .map((item) => {
      if (typeof item === "string") {
        return { id: item, label: item, expensive: false };
      }
      if (!item || typeof item !== "object" || typeof item.id !== "string") {
        return null;
      }
      return {
        id: item.id,
        label: String(item.label || item.name || item.id),
        expensive: Boolean(item.expensive),
      };
    })
    .filter(Boolean);
}

function fillAgentSelect(select, agents, preferred) {
  select.replaceChildren();
  for (const agent of agents) {
    const option = document.createElement("option");
    option.value = agent.id;
    const display = agent.label === agent.id ? agent.id : `${agent.label} · ${agent.id}`;
    option.textContent = agent.expensive ? `${display} · 较慢` : display;
    select.append(option);
  }
  if (agents.some((agent) => agent.id === preferred)) {
    select.value = preferred;
  }
}

function renderStartGameAvailability() {
  elements.startGameButton.disabled =
    app.busy || !app.agents.fugitive.length || !app.agents.marshal.length;
}

async function loadAgents() {
  renderStartGameAvailability();
  try {
    const payload = await requestJSON("/agents");
    app.agents.fugitive = normalizeAgentList(payload, "fugitive");
    app.agents.marshal = normalizeAgentList(payload, "marshal");
    fillAgentSelect(
      elements.fugitiveAgent,
      app.agents.fugitive,
      payload.defaults?.fugitive || "random",
    );
    fillAgentSelect(
      elements.marshalAgent,
      app.agents.marshal,
      payload.defaults?.marshal || "random",
    );
    renderStartGameAvailability();
  } catch (error) {
    showToast(error.message);
  }
}

function updateSetupMode() {
  const mode = getMode();
  const humanFugitive = mode === "human_fugitive";
  const humanMarshal = mode === "human_marshal";
  elements.fugitiveHuman.hidden = !humanFugitive;
  elements.fugitiveAgent.hidden = humanFugitive;
  elements.marshalHuman.hidden = !humanMarshal;
  elements.marshalAgent.hidden = humanMarshal;
  elements.marshalHuman.parentElement.classList.toggle("is-marshal", humanMarshal);
}

function parseOptionalSeed() {
  const raw = elements.gameSeed.value.trim();
  if (!raw) {
    return undefined;
  }
  if (raw.length > 20 || !/^(0|[1-9][0-9]*)$/.test(raw) || BigInt(raw) > MAX_SEED) {
    throw new APIError("随机种子必须是 0 到 2^64-1 的十进制整数", "invalid_seed", 0);
  }
  return raw;
}

async function startGame(event) {
  event.preventDefault();
  let seed;
  try {
    seed = parseOptionalSeed();
  } catch (error) {
    showToast(error.message);
    elements.gameSeed.focus();
    return;
  }

  const body = {
    mode: getMode(),
    fugitive_agent: elements.fugitiveAgent.value,
    marshal_agent: elements.marshalAgent.value,
  };
  if (seed !== undefined) {
    body.seed = seed;
  }

  setBusy(true);
  try {
    const game = await requestJSON("/games", { method: "POST", body });
    applyGameState(game);
    elements.setupPanel.hidden = true;
    elements.gameShell.hidden = false;
    elements.gameShell.scrollIntoView({ block: "start" });
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function resetSelections() {
  app.cardMode = "hideout";
  app.selectedHideout = null;
  app.selectedSprint.clear();
  app.fugitiveDraft = null;
  app.fugitiveDraftToken += 1;
  app.selectedGuesses.clear();
}

function selectionContext(game) {
  const hand = Array.isArray(game.hand) ? game.hand : [];
  const routeLength = Array.isArray(game.route) ? game.route.length : 0;
  const historyLength = Array.isArray(game.history) ? game.history.length : 0;
  return JSON.stringify([
    game.id,
    game.status,
    game.phase,
    game.actor,
    game.round_number,
    routeLength,
    historyLength,
    hand,
  ]);
}

function applyGameState(game) {
  if (!game || typeof game !== "object" || typeof game.id !== "string") {
    throw new APIError("服务器未返回有效对局状态", "invalid_game_state", 0);
  }
  const contextChanged = !app.game || selectionContext(app.game) !== selectionContext(game);
  app.game = game;
  app.gameId = game.id;
  if (contextChanged) {
    resetSelections();
  }
  if (["finished", "stalled", "terminated"].includes(game.status)) {
    stopAuto(false);
  }
  renderGame();
}

async function mutateGame(operation, body = {}) {
  if (!app.gameId || app.busy) {
    return null;
  }
  setBusy(true);
  try {
    const state = await requestJSON(`/games/${encodeURIComponent(app.gameId)}/${operation}`, {
      method: "POST",
      body,
    });
    applyGameState(state);
    return state;
  } catch (error) {
    showToast(error.message);
    throw error;
  } finally {
    setBusy(false);
  }
}

async function refreshGame() {
  if (!app.gameId || app.busy) {
    return;
  }
  try {
    const state = await requestJSON(`/games/${encodeURIComponent(app.gameId)}`);
    applyGameState(state);
  } catch (error) {
    showToast(error.message);
  }
}

function legalActions() {
  const value = app.game && app.game.legal_actions;
  return value && typeof value === "object" ? value : {};
}

function isHumanTurn() {
  if (!app.game || app.game.status !== "awaiting_human") {
    return false;
  }
  return app.game.viewer_role === app.game.actor;
}

function isFugitiveActionPhase() {
  return Boolean(
    app.game &&
      (app.game.phase === "fugitive_opening" || app.game.phase === "fugitive_action")
  );
}

function isMarshalGuessPhase() {
  return Boolean(
    app.game &&
      (app.game.phase === "marshal_guess" || app.game.phase === "manhunt")
  );
}

function renderGame() {
  if (!app.game) {
    return;
  }
  renderStatus();
  renderPiles();
  renderRoute();
  renderHand();
  renderActions();
  renderSession();
  renderHistory();
  renderSpectatorControls();
}

function renderStatus() {
  const game = app.game;
  elements.roundBadge.textContent = `回合 ${game.round_number ?? "—"}`;
  elements.phaseLabel.textContent = PHASE_LABELS[game.phase] || String(game.phase || "准备中");
  elements.sessionIdLabel.textContent = game.id ? `#${game.id.slice(0, 8)}` : "";

  if (game.status === "finished") {
    elements.turnLabel.textContent = "对局已经结束";
  } else if (game.status === "awaiting_human") {
    elements.turnLabel.textContent = `${ROLE_LABELS[game.actor] || game.actor} · 你的行动`;
  } else {
    elements.turnLabel.textContent = `${ROLE_LABELS[game.actor] || game.actor || "Agent"} · Agent 行动`;
  }

  const finished = game.status === "finished";
  elements.outcomeBanner.hidden = !finished;
  elements.outcomeBanner.classList.toggle("is-marshal", game.winner === "marshal");
  elements.outcomeBanner.classList.toggle("is-draw", game.winner === "draw");
  if (finished) {
    elements.outcomeTitle.textContent =
      game.winner === "draw"
        ? "对局平局"
        : `${ROLE_LABELS[game.winner] || game.winner} 获胜`;
    elements.outcomeReason.textContent = REASON_LABELS[game.reason] || String(game.reason || "");
  }
}

function renderPiles() {
  const sizes = Array.isArray(app.game.pile_sizes) ? app.game.pile_sizes : [];
  const legal = legalActions();
  const legalPiles = new Set(Array.isArray(legal.draw_piles) ? legal.draw_piles : []);
  elements.pileList.replaceChildren();

  for (let pile = 0; pile < 3; pile += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pile-button";
    button.dataset.pile = String(pile);
    button.classList.toggle("is-legal", legalPiles.has(pile));
    button.disabled = app.busy || !isHumanTurn() || !legalPiles.has(pile);
    button.setAttribute(
      "aria-label",
      `牌堆 ${pile + 1}，牌号 ${PILE_RANGES[pile]}，剩余 ${sizes[pile] ?? "未知"} 张`
    );

    const range = document.createElement("span");
    range.className = "pile-number";
    range.textContent = PILE_RANGES[pile];
    const count = document.createElement("strong");
    count.className = "pile-count";
    count.textContent = Number.isInteger(sizes[pile]) ? String(sizes[pile]) : "—";
    const label = document.createElement("span");
    label.className = "pile-label";
    label.textContent = `牌堆 ${pile + 1}`;
    button.append(range, count, label);
    button.addEventListener("click", () => {
      void mutateGame("action", { type: "draw", pile }).catch(() => {});
    });
    elements.pileList.append(button);
  }
}

function renderRoute() {
  const route = Array.isArray(app.game.route) ? app.game.route : [];
  elements.routeTrack.replaceChildren();
  elements.routeSummary.textContent = `${Math.max(0, route.length - 1)} 个藏身处`;

  if (!route.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "路线尚未建立";
    elements.routeTrack.append(empty);
    return;
  }

  for (const slot of route) {
    const index = Number.isInteger(slot.index) ? slot.index : 0;
    const hideout = Number.isInteger(slot.hideout) ? slot.hideout : null;
    const uncoveredByMarshal =
      slot.revealed === true && index > 0 && hideout !== 42;
    const node = document.createElement("div");
    node.className = "route-node";
    node.setAttribute("role", "listitem");

    const card = document.createElement("div");
    card.className = "route-card";
    card.setAttribute("role", "img");
    card.classList.toggle("is-hidden", hideout === null);
    card.classList.toggle("is-start", hideout === 0);
    card.classList.toggle("is-escape", hideout === 42);
    card.classList.toggle("is-revealed", uncoveredByMarshal);
    const cardDescription =
      hideout === null
        ? `第 ${index} 个隐藏藏身处`
        : `第 ${index} 个藏身处，牌号 ${hideout}`;
    card.setAttribute(
      "aria-label",
      uncoveredByMarshal ? `${cardDescription}，已被 Marshal 揭开` : cardDescription
    );

    const indexLabel = document.createElement("span");
    indexLabel.className = "route-index";
    indexLabel.textContent = index === 0 ? "START" : `#${index}`;
    const value = document.createElement("strong");
    value.className = hideout === null ? "hidden-pattern" : "route-value";
    value.textContent = hideout === null ? "?" : String(hideout);
    card.append(indexLabel, value);
    if (uncoveredByMarshal) {
      const badge = document.createElement("span");
      badge.className = "route-revealed-badge";
      badge.textContent = "已揭开";
      badge.setAttribute("aria-hidden", "true");
      card.append(badge);
    }
    node.append(card);

    const sprintStack = document.createElement("div");
    sprintStack.className = "sprint-stack";
    const sprintCards = Array.isArray(slot.sprint_cards) ? slot.sprint_cards : null;
    const sprintCount = Number.isInteger(slot.sprint_count) ? slot.sprint_count : 0;
    if (sprintCards !== null) {
      for (const sprintCard of sprintCards) {
        const token = document.createElement("span");
        token.className = "sprint-token";
        token.textContent = String(sprintCard);
        token.title = `Sprint 牌 ${sprintCard}`;
        sprintStack.append(token);
      }
    } else {
      for (let count = 0; count < sprintCount; count += 1) {
        const token = document.createElement("span");
        token.className = "sprint-token is-hidden";
        token.textContent = "S";
        token.title = "隐藏 Sprint 牌";
        sprintStack.append(token);
      }
    }
    node.append(sprintStack);
    elements.routeTrack.append(node);
  }
}

function cardNumbers(value) {
  return Array.isArray(value) ? value.filter(Number.isInteger) : [];
}

function handNumbers() {
  return app.game ? cardNumbers(app.game.hand) : [];
}

function displayedSprintValue(cardNumber) {
  if (cardNumber === 42) {
    return null;
  }
  return cardNumber % 2 === 0 ? 2 : 1;
}

function canSelectFugitiveCards() {
  return (
    !app.busy &&
    isHumanTurn() &&
    app.game.viewer_role === "fugitive" &&
    isFugitiveActionPhase()
  );
}

function canSelectFugitiveCard(cardNumber) {
  if (!canSelectFugitiveCards()) {
    return false;
  }
  const legal = legalActions();
  const field = app.cardMode === "hideout" ? "candidate_hideouts" : "sprint_cards";
  const allowed = new Set(cardNumbers(legal[field]));
  if (app.cardMode === "sprint" && !Number.isInteger(app.selectedHideout)) {
    return false;
  }
  return allowed.has(cardNumber) &&
    !(app.cardMode === "sprint" && app.selectedHideout === cardNumber);
}

function renderHand() {
  elements.handCards.replaceChildren();
  elements.handCards.classList.remove("is-dual");
  const game = app.game;
  const view = game.mode === "spectate" ? game.spectator_view : game.viewer_role;
  const hands = game.hands && typeof game.hands === "object" ? game.hands : {};

  if (game.mode === "spectate" && view === "omniscient") {
    const fugitiveHand = cardNumbers(hands.fugitive);
    const marshalHand = cardNumbers(hands.marshal);
    elements.handCards.classList.add("is-dual");
    elements.handSummary.textContent = `全知视角 · ${fugitiveHand.length + marshalHand.length} 张`;
    elements.handCards.append(
      createHandGroup("fugitive", fugitiveHand, false),
      createHandGroup("marshal", marshalHand, false)
    );
    return;
  }

  if (game.mode === "spectate" && view === "public") {
    elements.handSummary.textContent = "公开视角";
    appendHandEmptyState("公开视角不显示手牌");
    return;
  }

  const role = view === "fugitive" || view === "marshal" ? view : game.viewer_role;
  const hand = handNumbers();
  elements.handSummary.textContent = `${ROLE_LABELS[role] || "私有"} · ${hand.length} 张`;
  elements.handCards.append(
    createHandGroup(role, hand, role === "fugitive" && canSelectFugitiveCards())
  );
}

function appendHandEmptyState(message, parent = elements.handCards) {
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = message;
  parent.append(empty);
}

function createHandGroup(role, cards, interactive) {
  const group = document.createElement("section");
  group.className = "hand-group";
  group.classList.toggle("is-fugitive", role === "fugitive");
  group.classList.toggle("is-marshal", role === "marshal");
  group.setAttribute("aria-label", `${ROLE_LABELS[role] || role} 私有手牌`);

  const heading = document.createElement("div");
  heading.className = "hand-group-heading";
  const label = document.createElement("span");
  label.className = "hand-role-label";
  label.textContent = `${ROLE_LABELS[role] || role} 私有手牌`;
  const count = document.createElement("span");
  count.textContent = `${cards.length} 张`;
  heading.append(label, count);

  const row = document.createElement("div");
  row.className = "hand-card-row";
  row.setAttribute("role", "list");
  row.setAttribute("aria-label", `${ROLE_LABELS[role] || role} 手牌`);
  row.setAttribute("aria-description", "可横向滚动查看全部手牌");
  row.tabIndex = cards.length ? 0 : -1;
  if (!cards.length) {
    appendHandEmptyState("暂无手牌", row);
  } else {
    for (const cardNumber of cards) {
      row.append(createHandCard(cardNumber, interactive));
    }
  }
  group.append(heading, row);
  return group;
}

function createHandCard(cardNumber, interactive) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "hand-card";
  button.setAttribute("role", "listitem");
  button.setAttribute("aria-pressed", "false");
  const selectable = interactive && canSelectFugitiveCard(cardNumber);
  button.disabled = !selectable;
  button.classList.toggle("is-unavailable", interactive && !selectable);

  const isHideout = app.selectedHideout === cardNumber;
  const isSprint = app.selectedSprint.has(cardNumber);
  button.classList.toggle("is-hideout", isHideout);
  button.classList.toggle("is-sprint", isSprint);
  button.setAttribute("aria-pressed", String(isHideout || isSprint));
  const sprintValue = displayedSprintValue(cardNumber);
  button.setAttribute(
    "aria-label",
    `牌 ${cardNumber}${sprintValue === null ? "，逃脱牌" : `，Sprint 值 ${sprintValue}`}${
      isHideout ? "，选作藏身处" : isSprint ? "，选作 Sprint" : ""
    }`
  );

  const number = document.createElement("strong");
  number.className = "card-number";
  number.textContent = String(cardNumber);
  button.append(number);
  if (sprintValue !== null) {
    const sprint = document.createElement("span");
    sprint.className = "card-sprint";
    sprint.textContent = `+${sprintValue}`;
    button.append(sprint);
  }
  button.addEventListener("click", () => selectHandCard(cardNumber));
  return button;
}

function selectHandCard(cardNumber) {
  if (!canSelectFugitiveCard(cardNumber)) {
    return;
  }
  if (app.cardMode === "hideout") {
    app.selectedSprint.delete(cardNumber);
    app.selectedHideout = app.selectedHideout === cardNumber ? null : cardNumber;
  } else {
    if (app.selectedHideout === cardNumber) {
      app.selectedHideout = null;
    }
    if (app.selectedSprint.has(cardNumber)) {
      app.selectedSprint.delete(cardNumber);
    } else {
      app.selectedSprint.add(cardNumber);
    }
  }
  renderHand();
  void refreshFugitiveDraft();
}

function fugitiveDraftKey() {
  return JSON.stringify([
    app.gameId,
    app.selectedHideout,
    [...app.selectedSprint].sort((a, b) => a - b),
  ]);
}

async function refreshFugitiveDraft() {
  const token = app.fugitiveDraftToken + 1;
  app.fugitiveDraftToken = token;
  if (!Number.isInteger(app.selectedHideout)) {
    app.fugitiveDraft = null;
    renderFugitiveSelection();
    return;
  }

  const key = fugitiveDraftKey();
  const sprintCards = [...app.selectedSprint].sort((a, b) => a - b);
  app.fugitiveDraft = { key, status: "checking", canCommit: false };
  renderFugitiveSelection();
  try {
    const result = await requestJSON(
      `/games/${encodeURIComponent(app.gameId)}/preview`,
      {
        method: "POST",
        body: { hideout: app.selectedHideout, sprint_cards: sprintCards },
      },
    );
    if (token !== app.fugitiveDraftToken || key !== fugitiveDraftKey()) {
      return;
    }
    app.fugitiveDraft = {
      key,
      status: result.valid_prefix ? "valid" : "invalid",
      canCommit: Boolean(result.can_commit),
    };
  } catch (error) {
    if (token !== app.fugitiveDraftToken || key !== fugitiveDraftKey()) {
      return;
    }
    app.fugitiveDraft = { key, status: "error", canCommit: false };
    showToast(error.message);
  }
  renderFugitiveSelection();
}

function fugitiveSelectionStatus() {
  const legal = legalActions();
  const hideout = app.selectedHideout;
  const sprintCards = [...app.selectedSprint].sort((a, b) => a - b);
  const candidateHideouts = new Set(cardNumbers(legal.candidate_hideouts));
  const availableSprints = new Set(cardNumbers(legal.sprint_cards));

  if (!Number.isInteger(hideout)) {
    return {
      ready: false,
      error: false,
      label: "选择藏身处",
      sprintCards,
    };
  }
  if (!candidateHideouts.has(hideout)) {
    return {
      ready: false,
      error: true,
      label: "藏身处选择已失效",
      sprintCards,
    };
  }
  if (sprintCards.some((card) => card === hideout || !availableSprints.has(card))) {
    return {
      ready: false,
      error: true,
      label: "Sprint 选择已失效",
      sprintCards,
    };
  }
  const draft = app.fugitiveDraft;
  if (!draft || draft.key !== fugitiveDraftKey() || draft.status === "checking") {
    return { ready: false, error: false, label: "正在检查", sprintCards };
  }
  if (draft.status === "error") {
    return { ready: false, error: true, label: "无法检查当前组合", sprintCards };
  }
  if (draft.status === "invalid") {
    return { ready: false, error: true, label: "当前组合无法完成", sprintCards };
  }
  return {
    ready: draft.canCommit,
    error: false,
    label: draft.canCommit ? "可出牌" : "继续选择 Sprint",
    sprintCards,
  };
}

function setCardMode(mode) {
  app.cardMode = mode;
  for (const button of document.querySelectorAll("[data-card-mode]")) {
    const active = button.dataset.cardMode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function renderFugitiveSelection() {
  const status = fugitiveSelectionStatus();
  elements.selectedHideout.textContent = app.selectedHideout === null ? "—" : String(app.selectedHideout);
  elements.selectedSprint.textContent = `${status.sprintCards.length} 张`;
  elements.actionStatus.textContent = status.label;
  elements.actionStatus.classList.remove("is-valid");
  elements.actionStatus.classList.toggle("is-invalid", status.error);
  elements.playHideoutButton.disabled = app.busy || !status.ready;
  const canPass = Boolean(legalActions().can_pass);
  elements.passButton.disabled = app.busy || !canPass;
  elements.passButton.title = canPass ? "本回合 Pass" : "当前不可 Pass";
  setCardMode(app.cardMode);
}

function renderActions() {
  elements.drawAction.hidden = true;
  elements.fugitiveAction.hidden = true;
  elements.marshalAction.hidden = true;
  elements.waitingAction.hidden = true;
  elements.actionStatus.textContent = "";
  elements.actionStatus.classList.remove("is-valid", "is-invalid");

  if (app.game.status === "finished" || app.game.status === "terminated") {
    elements.waitingAction.hidden = false;
    elements.waitingText.textContent =
      app.game.status === "terminated" ? "会话已终止" : "对局结束";
    return;
  }

  if (app.game.status === "stalled") {
    elements.waitingAction.hidden = false;
    elements.waitingText.textContent = "自动推进达到安全上限";
    return;
  }

  if (app.game.mode === "spectate") {
    elements.waitingAction.hidden = false;
    elements.waitingText.textContent = app.autoRunning ? "自动播放中" : "观战已暂停";
    return;
  }

  if (!isHumanTurn()) {
    elements.waitingAction.hidden = false;
    elements.waitingText.textContent = "Agent 行动中";
    return;
  }

  const legal = legalActions();
  if (Array.isArray(legal.draw_piles) && legal.draw_piles.length) {
    elements.drawAction.hidden = false;
    elements.actionStatus.textContent = "从左侧选择牌堆";
    return;
  }
  if (app.game.viewer_role === "fugitive" && isFugitiveActionPhase()) {
    elements.fugitiveAction.hidden = false;
    renderFugitiveSelection();
    return;
  }
  if (app.game.viewer_role === "marshal" && isMarshalGuessPhase()) {
    elements.marshalAction.hidden = false;
    renderGuessGrid();
    return;
  }

  elements.waitingAction.hidden = false;
  elements.waitingText.textContent = "等待合法行动";
}

function renderGuessGrid() {
  const legal = legalActions();
  const allowed = new Set(Array.isArray(legal.guess_numbers) ? legal.guess_numbers : []);
  const maxCount = Number.isInteger(legal.max_guess_count) ? legal.max_guess_count : 0;
  const route = Array.isArray(app.game.route) ? app.game.route : [];
  const routeLength = Math.max(0, route.length - 1);
  const revealed = new Set(
    route
      .filter((slot) => slot && slot.revealed && Number.isInteger(slot.hideout))
      .map((slot) => slot.hideout),
  );
  const publicSprints = new Set(
    route.flatMap((slot) => (slot && Array.isArray(slot.sprint_cards) ? slot.sprint_cards : [])),
  );
  const marshalHand = new Set(
    app.game.viewer_role === "marshal" ? handNumbers() : [],
  );
  const failedSingletons = new Set(
    (Array.isArray(app.game.history) ? app.game.history : [])
      .filter(
        (event) =>
          event &&
          event.type === "guess" &&
          event.success === false &&
          event.route_length === routeLength &&
          Array.isArray(event.numbers) &&
          event.numbers.length === 1,
      )
      .map((event) => event.numbers[0]),
  );
  elements.guessGrid.replaceChildren();

  for (let number = 1; number <= 41; number += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "guess-number";
    button.textContent = String(number);
    const selected = app.selectedGuesses.has(number);
    button.classList.toggle("is-selected", selected);
    button.classList.toggle("is-revealed", revealed.has(number));
    button.classList.toggle(
      "is-known-out",
      marshalHand.has(number) || publicSprints.has(number),
    );
    button.classList.toggle("is-failed-singleton", failedSingletons.has(number));
    button.setAttribute("aria-pressed", String(selected));
    const annotations = [];
    if (revealed.has(number)) annotations.push("已揭示");
    if (marshalHand.has(number)) annotations.push("在你的手牌中");
    if (publicSprints.has(number)) annotations.push("公开 Sprint");
    if (failedSingletons.has(number)) annotations.push("当前路线长度下曾单猜失败");
    const annotation = annotations.length ? `，${annotations.join("，")}` : "";
    button.setAttribute("aria-label", `猜测 ${number}${annotation}`);
    button.title = annotations.join(" · ");
    button.disabled =
      app.busy ||
      !allowed.has(number) ||
      (!selected && maxCount > 0 && app.selectedGuesses.size >= maxCount);
    button.addEventListener("click", () => {
      if (app.selectedGuesses.has(number)) {
        app.selectedGuesses.delete(number);
      } else {
        app.selectedGuesses.add(number);
      }
      renderGuessGrid();
    });
    elements.guessGrid.append(button);
  }

  const minCount = Number.isInteger(legal.min_guess_count) ? legal.min_guess_count : 1;
  const count = app.selectedGuesses.size;
  const valid = count >= minCount && (maxCount === 0 || count <= maxCount);
  elements.guessCount.textContent = String(count);
  elements.clearGuessButton.disabled = app.busy || count === 0;
  elements.submitGuessButton.disabled = app.busy || !valid;
  elements.actionStatus.textContent =
    app.game.phase === "manhunt" ? "最终追捕 · 仅单张" : count ? `${count} 个号码 · 全中才揭开` : "选择猜测号码";
}

function renderSession() {
  const game = app.game;
  elements.modeValue.textContent = MODE_LABELS[game.mode] || String(game.mode || "—");
  elements.fugitiveValue.textContent =
    game.mode === "human_fugitive" ? "你 · Human" : String(game.fugitive_agent || "—");
  elements.marshalValue.textContent =
    game.mode === "human_marshal" ? "你 · Human" : String(game.marshal_agent || "—");
  elements.viewerValue.textContent =
    game.mode === "spectate"
      ? SPECTATOR_VIEW_LABELS[game.spectator_view] || "观战"
      : ROLE_LABELS[game.viewer_role] || String(game.viewer_role || "—");
  if (typeof game.seed === "string" && /^(0|[1-9][0-9]*)$/.test(game.seed)) {
    elements.seedValue.textContent = game.seed;
  } else if (game.seed_was_supplied) {
    elements.seedValue.textContent = "已指定 · 结束后显示";
  } else {
    elements.seedValue.textContent = "随机 · 结束后显示";
  }
  const exactSeedVisible = typeof game.seed === "string";
  elements.copySeedButton.hidden = !exactSeedVisible;
  elements.copySeedButton.disabled = app.busy || !exactSeedVisible;
  elements.exportTraceButton.hidden = !game.can_export;
  elements.exportTraceButton.disabled = app.busy || !game.can_export;
}

async function copyExactSeed() {
  if (!app.game || typeof app.game.seed !== "string") {
    return;
  }
  try {
    await navigator.clipboard.writeText(app.game.seed);
    showToast("Seed 已复制");
  } catch (_error) {
    showToast("浏览器未允许复制 Seed");
  }
}

async function exportTrace() {
  if (!app.gameId || !app.game?.can_export || app.busy) {
    return;
  }
  setBusy(true);
  try {
    const trace = await requestJSON(
      `/games/${encodeURIComponent(app.gameId)}/export`,
    );
    const blob = new Blob([JSON.stringify(trace, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `fugitive-${app.gameId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function canViewPrivateRole(role) {
  if (!app.game) {
    return false;
  }
  if (app.game.mode !== "spectate") {
    return app.game.viewer_role === role;
  }
  return app.game.spectator_view === "omniscient" || app.game.spectator_view === role;
}

function formatHistoryEvent(event) {
  const role = ROLE_LABELS[event.role] || String(event.role || "系统");
  const round = Number.isInteger(event.round_number) ? `回合 ${event.round_number}` : "SETUP";
  const type = String(event.type || "event");

  if (type === "draw") {
    const pile = Number.isInteger(event.pile) ? event.pile + 1 : "?";
    const ownVisibleCard = canViewPrivateRole(event.role) && Number.isInteger(event.card);
    return {
      role: event.role,
      meta: `${round} · ${role}`,
      text: ownVisibleCard
        ? `从牌堆 ${pile} 抽到 ${event.card}`
        : `从牌堆 ${pile} 抽牌`,
    };
  }

  if (type === "pass") {
    return { role: "fugitive", meta: `${round} · Fugitive`, text: "Pass" };
  }

  if (type === "fugitive_action") {
    const hideout = Number.isInteger(event.hideout) ? event.hideout : null;
    const sprintCount = Number.isInteger(event.sprint_count)
      ? event.sprint_count
      : Array.isArray(event.sprint_cards)
        ? event.sprint_cards.length
        : 0;
    const sprintCards = Array.isArray(event.sprint_cards)
      ? ` [${event.sprint_cards.join(", ")}]`
      : "";
    return {
      role: "fugitive",
      meta: `${round} · Fugitive`,
      text: `${hideout === null ? "建立隐藏藏身处" : `建立藏身处 ${hideout}`}${
        sprintCount ? ` · Sprint ${sprintCount}${sprintCards}` : ""
      }`,
    };
  }

  if (type === "guess") {
    const numbers = Array.isArray(event.numbers) ? event.numbers.join(", ") : "?";
    return {
      role: "marshal",
      meta: `${round} · Marshal${event.manhunt ? " · MANHUNT" : ""}`,
      text: `猜测 ${numbers} · ${event.success ? "命中" : "未命中"}`,
    };
  }

  return { role: null, meta: round, text: "状态更新" };
}

function renderHistory() {
  const history = Array.isArray(app.game.history) ? app.game.history : [];
  elements.gameLog.replaceChildren();
  elements.logCount.textContent = String(history.length);

  if (!history.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "暂无记录";
    elements.gameLog.append(empty);
    return;
  }

  for (const event of [...history].reverse()) {
    const formatted = formatHistoryEvent(event);
    const item = document.createElement("li");
    item.className = "log-entry";
    item.classList.toggle("is-fugitive", formatted.role === "fugitive");
    item.classList.toggle("is-marshal", formatted.role === "marshal");
    const meta = document.createElement("span");
    meta.className = "log-meta";
    meta.textContent = formatted.meta;
    const text = document.createElement("span");
    text.textContent = formatted.text;
    item.append(meta, text);
    elements.gameLog.append(item);
  }
}

function renderSpectatorControls() {
  const spectator = app.game && app.game.mode === "spectate";
  elements.spectatorControls.hidden = !spectator;
  if (!spectator) {
    return;
  }
  const stalled = app.game.status === "stalled";
  const finished = app.game.status === "finished" || app.game.status === "terminated";
  const currentView = Object.prototype.hasOwnProperty.call(
    SPECTATOR_VIEW_LABELS,
    app.game.spectator_view
  )
    ? app.game.spectator_view
    : "omniscient";
  elements.spectatorViewSelect.value = currentView;
  elements.spectatorViewSelect.disabled = app.busy;
  elements.stepButton.disabled = app.busy || finished || !app.game.can_step || app.autoRunning;
  elements.autoButton.disabled = app.busy || finished || !app.game.can_auto;
  elements.autoButton.hidden = stalled;
  elements.continueButton.hidden = !stalled;
  elements.terminateButton.hidden = !stalled;
  elements.continueButton.disabled = app.busy || !app.game.can_continue;
  elements.terminateButton.disabled = app.busy || !app.game.can_terminate;
  elements.autoButton.classList.toggle("is-running", app.autoRunning);
  elements.autoButton.title = app.autoRunning ? "暂停" : "自动播放";
  elements.autoButton.setAttribute("aria-label", app.autoRunning ? "暂停" : "自动播放");
  elements.autoState.textContent = app.autoRunning
    ? "播放中"
    : stalled
      ? "需处理"
      : finished
        ? app.game.status === "terminated" ? "已终止" : "已结束"
        : "已暂停";
  elements.autoState.classList.toggle("is-running", app.autoRunning);
  elements.speedSelect.disabled = app.busy || finished;
}

async function changeSpectatorView() {
  if (!app.gameId || app.game.mode !== "spectate" || app.busy) {
    return;
  }
  const requestedView = elements.spectatorViewSelect.value;
  const previousView = app.game.spectator_view || "omniscient";
  if (requestedView === previousView) {
    return;
  }
  try {
    await mutateGame("view", { spectator_view: requestedView });
  } catch (_error) {
    elements.spectatorViewSelect.value = previousView;
  }
}

function renderInteractiveState() {
  renderStartGameAvailability();
  if (!app.game) {
    return;
  }
  renderPiles();
  renderHand();
  renderActions();
  renderSpectatorControls();
  elements.resetButton.disabled = app.busy;
  elements.newGameButton.disabled = app.busy;
}

async function submitFugitiveAction() {
  const status = fugitiveSelectionStatus();
  if (!status.ready || app.selectedHideout === null) {
    return;
  }
  try {
    await mutateGame("action", {
      type: "fugitive_action",
      hideout: app.selectedHideout,
      sprint_cards: status.sprintCards,
    });
  } catch (_error) {
    // The server remains the final legality authority; retain the selection.
  }
}

async function submitGuess() {
  const numbers = [...app.selectedGuesses].sort((a, b) => a - b);
  if (!numbers.length) {
    return;
  }
  try {
    await mutateGame("action", { type: "guess", numbers });
  } catch (_error) {
    // Keep the selected numbers so the player can amend the action.
  }
}

async function stepSpectator() {
  stopAuto();
  try {
    await mutateGame("step", {});
  } catch (_error) {
    // Error is already shown by mutateGame.
  }
}

async function continueStalledGame() {
  stopAuto();
  try {
    await mutateGame("continue", {});
  } catch (_error) {
    // Error is already shown by mutateGame.
  }
}

async function terminateStalledGame() {
  if (!window.confirm("终止这个停滞的会话？规则层不会产生赢家。")) {
    return;
  }
  stopAuto();
  try {
    await mutateGame("terminate", {});
  } catch (_error) {
    // Error is already shown by mutateGame.
  }
}

function stopAuto(render = true) {
  app.autoRunning = false;
  window.clearTimeout(app.autoTimer);
  app.autoTimer = null;
  if (render && app.game) {
    renderActions();
    renderSpectatorControls();
  }
}

function scheduleAutoTick(immediate = false) {
  window.clearTimeout(app.autoTimer);
  if (!app.autoRunning) {
    return;
  }
  const delay = immediate ? 0 : Number(elements.speedSelect.value) || 850;
  app.autoTimer = window.setTimeout(() => void autoTick(), delay);
}

async function autoTick() {
  if (!app.autoRunning || !app.game || app.game.status === "finished") {
    if (app.game && app.game.status === "finished") {
      stopAuto();
    }
    return;
  }
  if (app.busy) {
    scheduleAutoTick();
    return;
  }
  try {
    await mutateGame("step", {});
  } catch (_error) {
    stopAuto();
    return;
  }
  if (app.autoRunning && app.game.can_auto) {
    scheduleAutoTick();
  } else {
    stopAuto();
  }
}

function toggleAuto() {
  if (app.autoRunning) {
    stopAuto();
    return;
  }
  if (!app.game || !app.game.can_auto) {
    return;
  }
  app.autoRunning = true;
  renderActions();
  renderSpectatorControls();
  scheduleAutoTick(true);
}

async function resetGame() {
  if (!app.gameId || !window.confirm("重置当前对局并重新洗牌？")) {
    return;
  }
  stopAuto();
  try {
    await mutateGame("reset", {});
  } catch (_error) {
    // Error is already shown by mutateGame.
  }
}

async function showNewGameSetup() {
  stopAuto();
  const previousGameId = app.gameId;
  app.game = null;
  app.gameId = null;
  resetSelections();
  elements.gameShell.hidden = true;
  elements.setupPanel.hidden = false;
  renderInteractiveState();
  elements.setupPanel.scrollIntoView({ block: "start" });
  if (previousGameId) {
    try {
      await requestJSON(`/games/${encodeURIComponent(previousGameId)}`, {
        method: "DELETE",
      });
    } catch (error) {
      showToast(error.message);
    }
  }
}

function bindEvents() {
  elements.setupForm.addEventListener("change", (event) => {
    if (event.target && event.target.name === "mode") {
      updateSetupMode();
    }
  });
  elements.setupForm.addEventListener("submit", startGame);
  elements.resetButton.addEventListener("click", () => void resetGame());
  elements.newGameButton.addEventListener("click", () => void showNewGameSetup());
  elements.passButton.addEventListener("click", () => {
    void mutateGame("action", { type: "pass" }).catch(() => {});
  });
  elements.playHideoutButton.addEventListener("click", () => void submitFugitiveAction());
  elements.clearGuessButton.addEventListener("click", () => {
    app.selectedGuesses.clear();
    renderGuessGrid();
  });
  elements.submitGuessButton.addEventListener("click", () => void submitGuess());
  elements.stepButton.addEventListener("click", () => void stepSpectator());
  elements.autoButton.addEventListener("click", toggleAuto);
  elements.continueButton.addEventListener("click", () => void continueStalledGame());
  elements.terminateButton.addEventListener("click", () => void terminateStalledGame());
  elements.copySeedButton.addEventListener("click", () => void copyExactSeed());
  elements.exportTraceButton.addEventListener("click", () => void exportTrace());
  elements.spectatorViewSelect.addEventListener("change", () => {
    void changeSpectatorView();
  });
  elements.speedSelect.addEventListener("change", () => {
    if (app.autoRunning) {
      scheduleAutoTick();
    }
  });
  for (const button of document.querySelectorAll("[data-card-mode]")) {
    button.addEventListener("click", () => {
      setCardMode(button.dataset.cardMode);
      renderHand();
    });
  }
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && app.gameId && !app.autoRunning) {
      void refreshGame();
    }
  });
}

async function initialize() {
  bindEvents();
  updateSetupMode();
  setConnection("connecting", "连接中");
  await loadAgents();
}

void initialize();
