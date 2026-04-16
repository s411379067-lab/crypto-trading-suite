const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const os = require("os");
const crypto = require("crypto");
const XLSX = require("xlsx");

let win;
let lastExcelPath = null;

// Use app-specific writable paths for Chromium cache/session files.
// This avoids Windows cache move/create errors (0x5) when default paths conflict.
const APP_DATA_DIR = path.join(app.getPath("appData"), "trade-journal-desktop");
const SESSION_DATA_DIR = path.join(APP_DATA_DIR, "session");

try {
  fs.mkdirSync(APP_DATA_DIR, { recursive: true });
  fs.mkdirSync(SESSION_DATA_DIR, { recursive: true });
  app.setPath("userData", APP_DATA_DIR);
  app.setPath("sessionData", SESSION_DATA_DIR);
} catch (e) {
  // Fallback to Electron defaults if custom path setup fails.
  console.warn("Failed to set custom app/session paths:", e?.message || e);
}

const DEFAULT_EXCEL_PATH = path.join(__dirname, 'data', 'trading_journal.xlsx');
// 或：path.join(__dirname, 'data', 'trading_journal.xlsx')

function getCustomizePlatformDir() {
  if (app.isPackaged) {
    return path.join(path.dirname(app.getPath("exe")), "customize platform");
  }
  return path.join(__dirname, "customize platform");
}

function ensureCustomizePlatformDir() {
  const dir = getCustomizePlatformDir();
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function sanitizePlatformName(inputName) {
  const raw = String(inputName || "").trim();
  const fallback = "platform_" + Date.now();
  const safe = (raw || fallback)
    .replace(/[<>:"/\\|?*]/g, "_")
    .replace(/\s+/g, " ")
    .trim();
  const capped = Array.from(safe || fallback).slice(0, 8).join("").trim();
  return capped || "platform";
}

function normalizePlatformFileName(inputName) {
  const name = sanitizePlatformName(inputName);
  return name.toLowerCase().endsWith(".json") ? name : `${name}.json`;
}

const AUTH_USERS = Object.freeze({
  admin: "admin123",
  demo: "demo123",
  jack: "jack77123",
  jack77: "jack77123",
});

const LICENSE_REQUIRED_USERS = new Set(["jack", "jack77"]);

function stableStringify(v) {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(stableStringify).join(",") + "]";
  const keys = Object.keys(v).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableStringify(v[k])).join(",") + "}";
}

function b64urlDecode(s) {
  const base64 = String(s).replace(/-/g, "+").replace(/_/g, "/");
  const padLen = (4 - (base64.length % 4)) % 4;
  return Buffer.from(base64 + "=".repeat(padLen), "base64");
}

function parseDateStrict(isoDate) {
  const s = String(isoDate || "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return null;
  const dt = new Date(`${s}T00:00:00.000Z`);
  return Number.isFinite(dt.getTime()) ? dt : null;
}

function parseIsoStrict(isoTs) {
  const dt = new Date(String(isoTs || ""));
  return Number.isFinite(dt.getTime()) ? dt : null;
}

function stateHmac(stateObj, machineId, secret) {
  const signed = stableStringify({ state: stateObj, machine_id: machineId });
  return crypto.createHmac("sha256", secret).update(signed).digest("hex");
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, obj) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(obj, null, 2), "utf8");
}

function utcTodayDate() {
  const now = new Date();
  const y = now.getUTCFullYear();
  const m = String(now.getUTCMonth() + 1).padStart(2, "0");
  const d = String(now.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function ensure(cond, msg) {
  if (!cond) throw new Error(msg);
}

function firstExistingPath(paths) {
  for (const p of paths) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function getMachineId() {
  const fromEnv = String(process.env.TJ_MACHINE_ID || "").trim();
  if (fromEnv) return fromEnv;
  return String(os.hostname() || "UNKNOWN-MACHINE").trim();
}

function verifyLicenseEnvelope(licenseObj, publicPem) {
  ensure(licenseObj && typeof licenseObj === "object", "License must be JSON object");
  ensure(licenseObj.meta && licenseObj.payload && licenseObj.signature, "License missing meta/payload/signature");
  ensure(licenseObj.meta.alg === "Ed25519", "Unsupported license alg");

  const signedText = stableStringify({ meta: licenseObj.meta, payload: licenseObj.payload });
  const ok = crypto.verify(
    null,
    Buffer.from(signedText, "utf8"),
    publicPem,
    b64urlDecode(String(licenseObj.signature))
  );
  ensure(ok, "License signature verification failed");
}

function validateLicensePayload(payload) {
  const required = [
    "license_id",
    "customer_id",
    "machine_id",
    "plan",
    "permissions",
    "issue_seq",
    "issued_at",
    "valid_from",
    "valid_to",
  ];
  required.forEach((k) => ensure(Object.prototype.hasOwnProperty.call(payload, k), `Missing payload.${k}`));
  ensure(Number.isInteger(payload.issue_seq) && payload.issue_seq > 0, "payload.issue_seq must be positive integer");
  ensure(parseIsoStrict(payload.issued_at), "payload.issued_at invalid");
  const from = parseDateStrict(payload.valid_from);
  const to = parseDateStrict(payload.valid_to);
  ensure(from && to, "payload.valid_from/valid_to must be YYYY-MM-DD");
  ensure(from.getTime() <= to.getTime(), "payload.valid_from must be <= payload.valid_to");
}

function loadState(statePath, machineId, stateSecret) {
  if (!fs.existsSync(statePath)) {
    return {
      max_issue_seq: 0,
      max_valid_from: "1970-01-01",
      max_issued_at: "1970-01-01T00:00:00.000Z",
      last_verified_at: "1970-01-01T00:00:00.000Z",
    };
  }

  const doc = readJson(statePath);
  ensure(doc && doc.state && doc.hmac, "Corrupted license state file");
  const expected = stateHmac(doc.state, machineId, stateSecret);
  ensure(expected === doc.hmac, "License state tampered");
  return doc.state;
}

function saveState(statePath, stateObj, machineId, stateSecret) {
  writeJson(statePath, {
    state: stateObj,
    hmac: stateHmac(stateObj, machineId, stateSecret),
  });
}

function verifyUserLicense(username) {
  const user = String(username || "").trim();
  ensure(user, "Invalid username for license verification");

  const licensePath = firstExistingPath([
    path.join(process.cwd(), "license", `${user}.lic.json`),
    path.join(APP_DATA_DIR, "license", `${user}.lic.json`),
    path.join(__dirname, "license", `${user}.lic.json`),
    path.join(__dirname, "license_system", "licenses", `${user}.lic.json`),
  ]);
  ensure(licensePath, `找不到 ${user} 授權檔，請放在 license/${user}.lic.json`);

  const publicKeyPath = firstExistingPath([
    path.join(process.cwd(), "license", "public_key.pem"),
    path.join(APP_DATA_DIR, "license", "public_key.pem"),
    path.join(__dirname, "license", "public_key.pem"),
    path.join(__dirname, "license_system", "keys", "public_key.pem"),
  ]);
  ensure(publicKeyPath, "找不到授權公鑰 public_key.pem");

  const licenseObj = readJson(licensePath);
  const publicPem = fs.readFileSync(publicKeyPath, "utf8");
  verifyLicenseEnvelope(licenseObj, publicPem);
  validateLicensePayload(licenseObj.payload);
  ensure(String(licenseObj.payload.customer_id || "").trim() === user, `License customer_id must be ${user}`);

  const now = new Date();
  const today = parseDateStrict(utcTodayDate());
  const from = parseDateStrict(licenseObj.payload.valid_from);
  const to = parseDateStrict(licenseObj.payload.valid_to);
  ensure(today.getTime() >= from.getTime(), "授權尚未生效");
  ensure(today.getTime() <= to.getTime(), "授權已過期");

  const machineId = getMachineId();
  const licenseMachine = String(licenseObj.payload.machine_id || "").trim();
  const machineMatch = licenseMachine === "*" || licenseMachine === machineId;
  ensure(machineMatch, `Machine mismatch: license=${licenseMachine}, local=${machineId}`);

  const stateSecret = String(process.env.LICENSE_STATE_SECRET || "dev-license-state-secret-change-me");
  const statePath = path.join(APP_DATA_DIR, "license", `${user}_license_state.json`);
  const state = loadState(statePath, machineId, stateSecret);

  const issuedAt = parseIsoStrict(licenseObj.payload.issued_at);
  const stateIssued = parseIsoStrict(state.max_issued_at);
  const stateFrom = parseDateStrict(state.max_valid_from);
  const stateLast = parseIsoStrict(state.last_verified_at);

  ensure(licenseObj.payload.issue_seq >= state.max_issue_seq, "Rollback detected: issue_seq smaller than local state");
  ensure(parseDateStrict(licenseObj.payload.valid_from).getTime() >= stateFrom.getTime(), "Rollback detected: valid_from older than local state");
  ensure(issuedAt.getTime() >= stateIssued.getTime(), "Rollback detected: issued_at older than local state");

  const CLOCK_SKEW_TOLERANCE_MS = 48 * 3600 * 1000;
  ensure(now.getTime() + CLOCK_SKEW_TOLERANCE_MS >= stateLast.getTime(), "Clock rollback detected");

  saveState(
    statePath,
    {
      max_issue_seq: Math.max(state.max_issue_seq || 0, licenseObj.payload.issue_seq),
      max_valid_from: parseDateStrict(licenseObj.payload.valid_from).getTime() > stateFrom.getTime()
        ? licenseObj.payload.valid_from
        : state.max_valid_from,
      max_issued_at: issuedAt.getTime() > stateIssued.getTime()
        ? licenseObj.payload.issued_at
        : state.max_issued_at,
      last_verified_at: now.toISOString(),
    },
    machineId,
    stateSecret
  );

  return {
    licensePath,
    machineId,
    plan: licenseObj.payload.plan,
    permissions: licenseObj.payload.permissions || {},
    validFrom: licenseObj.payload.valid_from,
    validTo: licenseObj.payload.valid_to,
  };
}

function createWindow() {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile("index.html");
}

app.whenReady().then(createWindow);

ipcMain.handle("auth:login", async (_evt, payload) => {
  const username = String(payload?.username || "").trim();
  const password = String(payload?.password || "");

  if (!username || !password) {
    return { ok: false, message: "請輸入帳號與密碼" };
  }

  if (!Object.prototype.hasOwnProperty.call(AUTH_USERS, username)) {
    return { ok: false, message: "帳號不存在" };
  }

  if (AUTH_USERS[username] !== password) {
    return { ok: false, message: "密碼錯誤" };
  }

  if (!LICENSE_REQUIRED_USERS.has(username)) {
    return {
      ok: true,
      user: username,
      plan: "basic",
      permissions: {},
      license: { mode: "bypass-non-license-user" },
    };
  }

  try {
    const verified = verifyUserLicense(username);
    return {
      ok: true,
      user: username,
      plan: verified.plan,
      permissions: verified.permissions,
      license: {
        mode: "signed-license",
        path: verified.licensePath,
        machine_id: verified.machineId,
        valid_from: verified.validFrom,
        valid_to: verified.validTo,
      },
    };
  } catch (e) {
    return { ok: false, message: `授權驗證失敗: ${e?.message || e}` };
  }
});

// --- 共用：確保檔案存在（不存在就建立一個空 Excel）---
function ensureExcelFileExists(filePath) {
  if (fs.existsSync(filePath)) return;

  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.json_to_sheet([]); // 空表
  XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  const out = XLSX.write(wb, { type: "buffer", bookType: "xlsx" });

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, out);
}

// --- 共用：讀取 Excel 並回傳 renderer 要的格式 ---
async function openExcelAtPath(filePath) {
  ensureExcelFileExists(filePath);

  const buf = fs.readFileSync(filePath);
  const wb = XLSX.read(buf, { type: "buffer" });
  const sheetName = wb.SheetNames[0] || "Sheet1";
  const ws = wb.Sheets[sheetName] || wb.Sheets[wb.SheetNames[0]];
  const rows = XLSX.utils.sheet_to_json(ws, { defval: "" });

  lastExcelPath = filePath;
  return { path: filePath, sheetName, rows };
}

function writeExcelAtPath(filePath, rows, sheetName) {
  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.json_to_sheet(rows || []);
  XLSX.utils.book_append_sheet(wb, ws, sheetName || "Sheet1");

  const out = XLSX.write(wb, { type: "buffer", bookType: "xlsx" });
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, out);
}

// --- IPC: 固定開預設檔 ---
ipcMain.handle("excel:openDefault", async () => {
  return await openExcelAtPath(DEFAULT_EXCEL_PATH);
});

// --- IPC: 指定路徑開檔 ---
ipcMain.handle("excel:openPath", async (_evt, filePath) => {
  if (!filePath || typeof filePath !== "string") throw new Error("Invalid path");
  return await openExcelAtPath(filePath);
});

// --- IPC: dialog 選檔 ---
ipcMain.handle("excel:open", async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(win, {
    properties: ["openFile"],
    filters: [{ name: "Excel", extensions: ["xlsx", "xls"] }],
  });
  if (canceled || !filePaths?.[0]) return null;
  return await openExcelAtPath(filePaths[0]);
});

// --- IPC: 存回同檔 ---
ipcMain.handle("excel:save", async (_evt, payload) => {
  const savePath = payload?.path || lastExcelPath;
  if (!savePath) throw new Error("尚未選擇 Excel 檔案");

  writeExcelAtPath(savePath, payload?.rows || [], payload?.sheetName || "Sheet1");

  lastExcelPath = savePath;
  return { ok: true, path: savePath };
});

// --- IPC: 另存新檔（讓 renderer 可選路徑）---
ipcMain.handle("excel:saveAs", async (_evt, payload) => {
  const defaultPath = payload?.path || lastExcelPath || DEFAULT_EXCEL_PATH;
  const { canceled, filePath } = await dialog.showSaveDialog(win, {
    defaultPath,
    filters: [{ name: "Excel", extensions: ["xlsx"] }],
  });

  if (canceled || !filePath) {
    return { ok: false, canceled: true };
  }

  const targetPath = String(filePath).toLowerCase().endsWith(".xlsx")
    ? filePath
    : `${filePath}.xlsx`;

  writeExcelAtPath(targetPath, payload?.rows || [], payload?.sheetName || "Sheet1");
  lastExcelPath = targetPath;
  return { ok: true, path: targetPath };
});

ipcMain.handle("customize:listPlatforms", async () => {
  const dir = ensureCustomizePlatformDir();
  const files = fs.readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".json"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" }));

  return {
    ok: true,
    dir,
    files: files.map((fileName) => ({
      fileName,
      name: fileName.replace(/\.json$/i, ""),
    })),
  };
});

ipcMain.handle("customize:savePlatform", async (_evt, payload) => {
  const dir = ensureCustomizePlatformDir();
  const fileName = normalizePlatformFileName(payload?.name);
  const targetPath = path.join(dir, fileName);

  const platformData = {
    savedAt: new Date().toISOString(),
    data: payload?.data ?? {},
  };

  fs.writeFileSync(targetPath, JSON.stringify(platformData, null, 2), "utf8");

  return {
    ok: true,
    dir,
    fileName,
    name: fileName.replace(/\.json$/i, ""),
    path: targetPath,
  };
});

ipcMain.handle("customize:savePlatformAs", async (_evt, payload) => {
  const dir = ensureCustomizePlatformDir();
  const defaultName = normalizePlatformFileName(payload?.defaultName || "platform");
  const defaultPath = path.join(dir, defaultName);

  const { canceled, filePath } = await dialog.showSaveDialog(win, {
    defaultPath,
    filters: [{ name: "Customize Platform", extensions: ["json"] }],
  });

  if (canceled || !filePath) {
    return { ok: false, canceled: true };
  }

  const chosenDir = path.dirname(filePath);
  const chosenBase = path.basename(filePath, path.extname(filePath));
  const fileName = normalizePlatformFileName(chosenBase);
  const targetPath = path.join(chosenDir, fileName);

  const platformData = {
    savedAt: new Date().toISOString(),
    data: payload?.data ?? {},
  };

  fs.writeFileSync(targetPath, JSON.stringify(platformData, null, 2), "utf8");

  return {
    ok: true,
    dir,
    fileName,
    name: fileName.replace(/\.json$/i, ""),
    path: targetPath,
  };
});

ipcMain.handle("customize:loadPlatform", async (_evt, payload) => {
  const dir = ensureCustomizePlatformDir();
  const fileName = normalizePlatformFileName(payload?.name);
  const targetPath = path.join(dir, fileName);

  if (!fs.existsSync(targetPath)) {
    return { ok: false, message: `Platform not found: ${fileName}` };
  }

  const raw = fs.readFileSync(targetPath, "utf8");
  const parsed = JSON.parse(raw || "{}");

  return {
    ok: true,
    dir,
    fileName,
    name: fileName.replace(/\.json$/i, ""),
    data: parsed?.data ?? {},
    savedAt: parsed?.savedAt || null,
  };
});
