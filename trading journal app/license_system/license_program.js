#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");

// -----------------------------
// Helpers
// -----------------------------
function die(msg, code = 1) {
  console.error(`[ERROR] ${msg}`);
  process.exit(code);
}

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function writeText(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, text, "utf8");
}

function readJson(filePath) {
  return JSON.parse(readText(filePath));
}

function writeJson(filePath, obj) {
  writeText(filePath, JSON.stringify(obj, null, 2));
}

function b64urlEncode(buf) {
  return Buffer.from(buf)
    .toString("base64")
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

function b64urlDecode(s) {
  const base64 = String(s).replace(/-/g, "+").replace(/_/g, "/");
  const padLen = (4 - (base64.length % 4)) % 4;
  return Buffer.from(base64 + "=".repeat(padLen), "base64");
}

function stableStringify(v) {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(stableStringify).join(",") + "]";
  const keys = Object.keys(v).sort();
  return (
    "{" +
    keys.map((k) => JSON.stringify(k) + ":" + stableStringify(v[k])).join(",") +
    "}"
  );
}

function sha256Hex(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
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

function ensure(cond, msg) {
  if (!cond) throw new Error(msg);
}

function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (cur.startsWith("--")) {
      const key = cur.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        out[key] = next;
        i += 1;
      } else {
        out[key] = true;
      }
    } else {
      out._.push(cur);
    }
  }
  return out;
}

function utcTodayDate() {
  const now = new Date();
  const y = now.getUTCFullYear();
  const m = String(now.getUTCMonth() + 1).padStart(2, "0");
  const d = String(now.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

// -----------------------------
// License format
// -----------------------------
function buildLicense(payload, privatePem, kid = "main") {
  const env = {
    meta: {
      format_version: 1,
      alg: "Ed25519",
      kid,
      typ: "TRADING_JOURNAL_LICENSE"
    },
    payload
  };

  const signedText = stableStringify({ meta: env.meta, payload: env.payload });
  const signature = crypto.sign(null, Buffer.from(signedText, "utf8"), privatePem);
  env.signature = b64urlEncode(signature);
  return env;
}

function verifyLicenseEnvelope(licenseObj, publicPem) {
  ensure(licenseObj && typeof licenseObj === "object", "License must be a JSON object");
  ensure(licenseObj.meta && licenseObj.payload && licenseObj.signature, "License missing meta/payload/signature");
  ensure(licenseObj.meta.alg === "Ed25519", "Unsupported alg, expected Ed25519");

  const signedText = stableStringify({ meta: licenseObj.meta, payload: licenseObj.payload });
  const ok = crypto.verify(
    null,
    Buffer.from(signedText, "utf8"),
    publicPem,
    b64urlDecode(String(licenseObj.signature))
  );

  if (!ok) throw new Error("Signature verification failed");
  return true;
}

function validatePayloadShape(payload) {
  ensure(payload, "Payload required");

  const required = [
    "license_id",
    "customer_id",
    "machine_id",
    "plan",
    "permissions",
    "issue_seq",
    "issued_at",
    "valid_from",
    "valid_to"
  ];

  for (const k of required) {
    ensure(Object.prototype.hasOwnProperty.call(payload, k), `Missing payload.${k}`);
  }

  ensure(typeof payload.license_id === "string" && payload.license_id.length > 0, "payload.license_id must be non-empty string");
  ensure(typeof payload.customer_id === "string" && payload.customer_id.length > 0, "payload.customer_id must be non-empty string");
  ensure(typeof payload.machine_id === "string" && payload.machine_id.length > 0, "payload.machine_id must be non-empty string");
  ensure(typeof payload.plan === "string" && payload.plan.length > 0, "payload.plan must be non-empty string");
  ensure(payload.permissions && typeof payload.permissions === "object", "payload.permissions must be object");
  ensure(Number.isInteger(payload.issue_seq) && payload.issue_seq > 0, "payload.issue_seq must be positive integer");

  const issuedAt = parseIsoStrict(payload.issued_at);
  ensure(issuedAt, "payload.issued_at must be valid ISO timestamp");

  const from = parseDateStrict(payload.valid_from);
  const to = parseDateStrict(payload.valid_to);
  ensure(from && to, "payload.valid_from/valid_to must be YYYY-MM-DD");
  ensure(from.getTime() <= to.getTime(), "payload.valid_from must be <= payload.valid_to");

  if (payload.not_before) {
    ensure(parseIsoStrict(payload.not_before), "payload.not_before must be valid ISO timestamp");
  }
  if (payload.not_after) {
    ensure(parseIsoStrict(payload.not_after), "payload.not_after must be valid ISO timestamp");
  }
}

// -----------------------------
// Local anti-rollback state
// -----------------------------
function stateHmac(stateObj, machineId, secret) {
  const signed = stableStringify({ state: stateObj, machine_id: machineId });
  return crypto.createHmac("sha256", secret).update(signed).digest("hex");
}

function defaultStatePath() {
  const appData = process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming");
  return path.join(appData, "TradingJournal", "license_state.json");
}

function loadState(statePath, machineId, stateSecret) {
  if (!fs.existsSync(statePath)) {
    return {
      max_issue_seq: 0,
      max_valid_from: "1970-01-01",
      max_issued_at: "1970-01-01T00:00:00.000Z",
      max_license_hash: "",
      last_verified_at: "1970-01-01T00:00:00.000Z"
    };
  }

  const doc = readJson(statePath);
  ensure(doc && typeof doc === "object" && doc.state && doc.hmac, "Corrupted state file structure");

  const expected = stateHmac(doc.state, machineId, stateSecret);
  ensure(expected === doc.hmac, "State file HMAC mismatch (state tampered)");

  return doc.state;
}

function saveState(statePath, stateObj, machineId, stateSecret) {
  const doc = {
    state: stateObj,
    hmac: stateHmac(stateObj, machineId, stateSecret)
  };
  writeJson(statePath, doc);
}

function enforceAntiRollback(payload, now, stateObj) {
  const from = parseDateStrict(payload.valid_from);
  const stFrom = parseDateStrict(stateObj.max_valid_from);
  const issuedAt = parseIsoStrict(payload.issued_at);
  const stIssuedAt = parseIsoStrict(stateObj.max_issued_at);
  const stVerifiedAt = parseIsoStrict(stateObj.last_verified_at);

  ensure(payload.issue_seq >= stateObj.max_issue_seq, `Rollback detected: issue_seq ${payload.issue_seq} < stored ${stateObj.max_issue_seq}`);
  ensure(from.getTime() >= stFrom.getTime(), `Rollback detected: valid_from ${payload.valid_from} < stored ${stateObj.max_valid_from}`);
  ensure(issuedAt.getTime() >= stIssuedAt.getTime(), `Rollback detected: issued_at ${payload.issued_at} < stored ${stateObj.max_issued_at}`);

  const CLOCK_SKEW_TOLERANCE_MS = 48 * 3600 * 1000;
  ensure(
    now.getTime() + CLOCK_SKEW_TOLERANCE_MS >= stVerifiedAt.getTime(),
    `Clock rollback detected: now ${now.toISOString()} is too far behind last_verified_at ${stateObj.last_verified_at}`
  );
}

function updateStateAfterSuccess(payload, now, stateObj, licenseObj) {
  const licenseHash = sha256Hex(stableStringify(licenseObj));
  return {
    max_issue_seq: Math.max(stateObj.max_issue_seq || 0, payload.issue_seq),
    max_valid_from: (parseDateStrict(payload.valid_from).getTime() > parseDateStrict(stateObj.max_valid_from).getTime())
      ? payload.valid_from
      : stateObj.max_valid_from,
    max_issued_at: (parseIsoStrict(payload.issued_at).getTime() > parseIsoStrict(stateObj.max_issued_at).getTime())
      ? payload.issued_at
      : stateObj.max_issued_at,
    max_license_hash: licenseHash,
    last_verified_at: now.toISOString()
  };
}

function enforceDateWindow(payload, now) {
  const today = parseDateStrict(utcTodayDate());
  const from = parseDateStrict(payload.valid_from);
  const to = parseDateStrict(payload.valid_to);

  ensure(today.getTime() >= from.getTime(), `License not active yet: valid_from=${payload.valid_from}`);
  ensure(today.getTime() <= to.getTime(), `License expired: valid_to=${payload.valid_to}`);

  if (payload.not_before) {
    const nb = parseIsoStrict(payload.not_before);
    ensure(now.getTime() >= nb.getTime(), `License blocked before ${payload.not_before}`);
  }
  if (payload.not_after) {
    const na = parseIsoStrict(payload.not_after);
    ensure(now.getTime() <= na.getTime(), `License blocked after ${payload.not_after}`);
  }
}

// -----------------------------
// Commands
// -----------------------------
function cmdGenKeys(args) {
  const outDir = args.out || path.join(process.cwd(), "keys");
  fs.mkdirSync(outDir, { recursive: true });

  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519");
  const priPem = privateKey.export({ type: "pkcs8", format: "pem" });
  const pubPem = publicKey.export({ type: "spki", format: "pem" });

  writeText(path.join(outDir, "private_key.pem"), priPem);
  writeText(path.join(outDir, "public_key.pem"), pubPem);

  console.log(`[OK] Keys generated in ${outDir}`);
}

function cmdIssue(args) {
  const inFile = args.in;
  const outFile = args.out;
  const privateFile = args.private;
  const kid = args.kid || "main";

  if (!inFile || !outFile || !privateFile) {
    die("Usage: node license_program.js issue --in license_input.json --private private_key.pem --out license.lic.json [--kid 2026-04]");
  }

  const payload = readJson(inFile);
  validatePayloadShape(payload);

  const privatePem = readText(privateFile);
  const lic = buildLicense(payload, privatePem, kid);
  writeJson(outFile, lic);

  console.log(`[OK] License issued: ${outFile}`);
  console.log(`[INFO] license_id=${payload.license_id}, issue_seq=${payload.issue_seq}, valid=${payload.valid_from}..${payload.valid_to}`);
}

function cmdVerify(args) {
  const licFile = args.license;
  const publicFile = args.public;
  const machineId = args["machine-id"];
  const statePath = args.state || defaultStatePath();

  if (!licFile || !publicFile || !machineId) {
    die("Usage: node license_program.js verify --license xxx.lic.json --public public_key.pem --machine-id MACHINE123 [--state path]");
  }

  const now = args.now ? new Date(args.now) : new Date();
  if (!Number.isFinite(now.getTime())) die("--now must be valid ISO timestamp");

  const stateSecret = process.env.LICENSE_STATE_SECRET;
  if (!stateSecret || String(stateSecret).length < 16) {
    die("Set env LICENSE_STATE_SECRET (at least 16 chars) before verify");
  }

  const publicPem = readText(publicFile);
  const lic = readJson(licFile);

  verifyLicenseEnvelope(lic, publicPem);
  validatePayloadShape(lic.payload);

  ensure(lic.payload.machine_id === machineId, `Machine mismatch: license=${lic.payload.machine_id}, local=${machineId}`);

  enforceDateWindow(lic.payload, now);

  const state = loadState(statePath, machineId, stateSecret);
  enforceAntiRollback(lic.payload, now, state);

  const newState = updateStateAfterSuccess(lic.payload, now, state, lic);
  saveState(statePath, newState, machineId, stateSecret);

  const permissions = lic.payload.permissions || {};
  console.log("[OK] License verified");
  console.log(JSON.stringify({
    customer_id: lic.payload.customer_id,
    plan: lic.payload.plan,
    issue_seq: lic.payload.issue_seq,
    valid_from: lic.payload.valid_from,
    valid_to: lic.payload.valid_to,
    permissions
  }, null, 2));
}

function cmdHelp() {
  console.log(`
Trading Journal License Program

Commands:
  gen-keys
    node license_program.js gen-keys --out ./keys

  issue
    node license_program.js issue \
      --in ./license_input.json \
      --private ./keys/private_key.pem \
      --out ./licenses/CUST001-2026-04.lic.json \
      --kid 2026-04

  verify
    set LICENSE_STATE_SECRET=your-long-local-secret
    node license_program.js verify \
      --license ./licenses/CUST001-2026-04.lic.json \
      --public ./keys/public_key.pem \
      --machine-id CUST001-MACHINE-A \
      --state "%APPDATA%/TradingJournal/license_state.json"

Notes:
  - Signature uses Ed25519.
  - Anti-rollback uses local state with HMAC tamper detection.
  - For real product security, keep private key offline/HSM and rotate key by kid.
`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const cmd = args._[0];

  try {
    if (!cmd || cmd === "help" || cmd === "--help" || cmd === "-h") return cmdHelp();
    if (cmd === "gen-keys") return cmdGenKeys(args);
    if (cmd === "issue") return cmdIssue(args);
    if (cmd === "verify") return cmdVerify(args);
    die(`Unknown command: ${cmd}`);
  } catch (err) {
    die(err && err.message ? err.message : String(err));
  }
}

main();
