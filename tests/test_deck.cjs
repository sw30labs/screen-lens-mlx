// Exercise the real deck script with local DOM/API fakes; no model or browser needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../src/web/static/deck.js"), "utf8");
const sparkUrl = "http://sparkone.local:8000/v1";
const sparkModel = "qwen3.8-flash-next";
const online = (model = sparkModel) => ({
  reachable: true, provider: "vllm", base_url: sparkUrl,
  models: [model], vision_models: [model], text_models: [],
});
const offline = {
  reachable: false, provider: "vllm", base_url: sparkUrl, models: [], detail: "timed out",
};
const roles = (model = sparkModel) => ({
  caption: {model, provider: "vllm", base_url: sparkUrl, vision_ok: true},
  text: {model, provider: "vllm", base_url: sparkUrl},
});
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};

async function deck(initial = online()) {
  const elements = new Map();
  function element(selector) {
    if (!elements.has(selector)) elements.set(selector, {
      value: "", textContent: "", style: {}, dataset: {}, listeners: {},
      classList: {add() {}, remove() {}, toggle() {}},
      querySelectorAll() { return []; },
      addEventListener(name, fn) { this.listeners[name] = fn; },
      set innerHTML(html) {
        this.html = html;
        if (["#f-vision", "#f-text"].includes(selector)) {
          this.value = html.match(/<option value="([^"]*)"/)?.[1] || "";
        }
      },
      get innerHTML() { return this.html || ""; },
    });
    return elements.get(selector);
  }
  element("#f-backend").value = "vllm";
  element("#f-baseurl").value = sparkUrl;
  element("#f-pipeline").value = "ingest";
  element("#f-video").value = "/tmp/demo.mov";
  const alerts = [], requests = [];
  const app = {
    element, alerts, requests,
    backend: async () => initial,
    roles: async () => roles(),
  };
  const context = vm.createContext({
    document: {querySelector: element, querySelectorAll: () => []},
    location: {hash: ""}, URLSearchParams, console, setInterval() {},
    alert: message => alerts.push(message),
    fetch: async (url, opts) => {
      requests.push({url, opts});
      let data = {};
      if (url.startsWith("/api/backend")) data = await app.backend(url);
      else if (url.startsWith("/api/roles")) data = await app.roles(url);
      else if (url === "/api/runs") data = {runs: []};
      else if (url === "/api/health") data = {busy: false};
      return {ok: true, json: async () => data};
    },
  });
  app.run = code => vm.runInContext(code, context);
  await app.run(source); // Includes normal page initialization and event wiring.
  return app;
}

test("Spark preset discovers the served vision model and submits that endpoint", async () => {
  const app = await deck();
  app.element("#f-baseurl").value = "http://old-ip:8000/v1";
  app.element("#f-backend").listeners.change();
  assert.equal(app.element("#f-baseurl").value, sparkUrl);
  await app.run("loadBackend()");
  assert.equal(app.element("#f-vision").value, sparkModel);
  await app.run("startRun()");
  const request = app.requests.find(r => r.url === "/api/run");
  const body = JSON.parse(request.opts.body);
  assert.equal(body.backend, "vllm");
  assert.equal(body.base_url, sparkUrl);
  assert.equal(body.vision_model, sparkModel);
  assert.equal(body.text_model, sparkModel);
  assert.deepEqual(app.alerts, []);
});

test("an offline endpoint reports the connection failure and can be retried", async () => {
  const app = await deck(offline);
  await app.run("startRun()");
  assert.match(app.alerts[0], /unreachable.*sparkone\.local.*timed out/);
  assert.doesNotMatch(app.alerts[0], /no recognized vision/);
  assert.equal(app.element("#f-vision").value, "");
  assert.equal(app.requests.some(r => r.url === "/api/run"), false);
  app.backend = async () => online();
  await app.element("#backend-retry").listeners.click();
  assert.equal(app.element("#f-vision").value, sparkModel);
  await app.run("startRun()");
  assert.equal(app.requests.some(r => r.url === "/api/run"), true);
});

test("switching endpoints clears old models and blocks submission during discovery", async () => {
  const app = await deck();
  const pending = deferred();
  app.backend = () => pending.promise;
  app.element("#f-baseurl").value = "http://new-host:8000/v1";
  const loading = app.run("loadBackend()");
  assert.equal(app.element("#f-vision").value, "");
  assert.equal(app.element("#f-text").value, "");
  await app.run("startRun()");
  assert.match(app.alerts[0], /Wait for the selected endpoint check/);
  assert.equal(app.requests.some(r => r.url === "/api/run"), false);
  pending.resolve(online("new-vision"));
  await loading;
  assert.equal(app.element("#f-vision").value, "new-vision");
});

test("a late failed probe cannot overwrite the newly selected endpoint", async () => {
  const app = await deck();
  const old = deferred();
  app.backend = () => old.promise;
  const loading = app.run("loadBackend()");
  app.backend = async () => online("current-vision");
  await app.run("loadBackend()");
  old.reject(new Error("old endpoint timed out"));
  await loading;
  assert.equal(app.element("#f-vision").value, "current-vision");
  assert.match(app.element("#backend-status").innerHTML, /reachable/);
  assert.equal(app.run("ENDPOINT.reachable"), true);
});

test("late role responses cannot overwrite current models or badges", async () => {
  const app = await deck();
  const old = deferred(), requested = deferred();
  app.roles = () => { requested.resolve(); return old.promise; };
  const loading = app.run("loadBackend()");
  await requested.promise;
  app.backend = async () => online("current-vision");
  app.roles = async () => roles("current-vision");
  await app.run("loadBackend()");
  old.resolve(roles("old-vision"));
  await loading;
  assert.equal(app.element("#f-vision").value, "current-vision");
  assert.equal(app.element("#badge-vision").textContent, "◉ current-vision");
});

test("a reachable text-only server reports missing vision separately", async () => {
  const app = await deck({...online("deepseek-v4-flash"), vision_models: []});
  await app.run("startRun()");
  assert.match(app.alerts[0], /reachable but has no recognized vision model/);
  assert.equal(app.requests.some(r => r.url === "/api/run"), false);
});

test("editing a checked URL requires discovery before submission", async () => {
  const app = await deck();
  app.element("#f-baseurl").value = "http://different:8000/v1";
  await app.run("startRun()");
  assert.match(app.alerts[0], /selected endpoint check/);
  assert.equal(app.requests.some(r => r.url === "/api/run"), false);
});

test("editing the URL during a pending probe does not validate the unprobed URL", async () => {
  const app = await deck();
  const pending = deferred();
  app.backend = () => pending.promise;
  const loading = app.run("loadBackend()");
  app.element("#f-baseurl").value = "http://different:8000/v1";
  pending.resolve(online());
  await loading;
  await app.run("startRun()");
  assert.equal(app.element("#f-vision").value, "");
  assert.equal(app.requests.some(r => r.url === "/api/run"), false);
});
