const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
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

  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.json_to_sheet(payload.rows || []);
  XLSX.utils.book_append_sheet(wb, ws, payload.sheetName || "Sheet1");

  const out = XLSX.write(wb, { type: "buffer", bookType: "xlsx" });
  fs.writeFileSync(savePath, out);

  lastExcelPath = savePath;
  return { ok: true, path: savePath };
});
