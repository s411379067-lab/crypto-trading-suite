# 離線月授權方案（簽章 + 驗簽 + 防回滾）

這個資料夾提供一個可直接使用的 **新程式**：`license_program.js`。

可處理：
- 每月簽發一份授權檔
- 客戶端本機驗簽（離線）
- 偵測授權檔或本機狀態被竄改
- 防止使用者把授權回滾到舊月份

## 1. 權限檔格式

權限檔（例如 `CUST001-2026-04.lic.json`）是這個結構：

```json
{
  "meta": {
    "format_version": 1,
    "alg": "Ed25519",
    "kid": "2026-04",
    "typ": "TRADING_JOURNAL_LICENSE"
  },
  "payload": {
    "license_id": "LIC-CUST001-2026-04",
    "customer_id": "CUST001",
    "machine_id": "CUST001-MACHINE-A",
    "plan": "pro",
    "permissions": {
      "can_open_excel": true,
      "can_save_excel": true,
      "can_use_analytics": true,
      "can_use_charts": true,
      "max_rows": 50000
    },
    "issue_seq": 202604,
    "issued_at": "2026-04-01T00:00:00.000Z",
    "valid_from": "2026-04-01",
    "valid_to": "2026-04-30",
    "not_before": "2026-04-01T00:00:00.000Z",
    "not_after": "2026-05-03T00:00:00.000Z"
  },
  "signature": "base64url-encoded-ed25519-signature"
}
```

說明：
- `issue_seq`：每月遞增序號（例如 202604）
- `valid_from`/`valid_to`：授權生效期間
- `machine_id`：綁定客戶機器
- `signature`：對 `meta + payload` 做 Ed25519 簽章

## 2. 驗簽流程

程式 `verify` 命令實作了完整流程：

1. 讀取權限檔 JSON
2. 用公鑰驗證 `signature`
3. 檢查 payload 欄位與日期格式
4. 檢查 `machine_id` 是否與本機一致
5. 檢查授權是否在有效期
6. 載入本機狀態檔並驗 HMAC（防本機狀態被改）
7. 執行防回滾規則
8. 成功後更新本機狀態檔

## 3. 本機防回滾規則

本機保存 `license_state.json`（預設 `%APPDATA%/TradingJournal/license_state.json`），裡面保存：
- `max_issue_seq`
- `max_valid_from`
- `max_issued_at`
- `last_verified_at`
- `max_license_hash`

每次驗證都會拒絕以下情況：
- 新授權 `issue_seq` 小於歷史最大值
- 新授權 `valid_from` 早於歷史最大值
- 新授權 `issued_at` 早於歷史最大值
- 系統時間明顯回撥（超過 48 小時容忍）

狀態檔附帶 HMAC：
- HMAC key 使用環境變數 `LICENSE_STATE_SECRET`
- 若狀態檔被改動，HMAC 會不一致，程式拒絕

## 4. 使用方式

### A. 產生金鑰（只做一次）

```bash
node license_program.js gen-keys --out ./keys
```

輸出：
- `keys/private_key.pem`（只在你方保存）
- `keys/public_key.pem`（放進客戶端 exe）

### B. 每月簽發授權檔

1. 複製 `license_input.sample.json` 成你的客戶月份資料
2. 執行簽發：

```bash
node license_program.js issue \
  --in ./license_input.sample.json \
  --private ./keys/private_key.pem \
  --out ./licenses/CUST001-2026-04.lic.json \
  --kid 2026-04
```

### C. 客戶端離線驗簽

```bash
set LICENSE_STATE_SECRET=replace-with-strong-secret
node license_program.js verify \
  --license ./licenses/CUST001-2026-04.lic.json \
  --public ./keys/public_key.pem \
  --machine-id CUST001-MACHINE-A
```

## 5. 你在 Electron App 的整合方式

在 app 啟動時：
- 讀取客戶放置的 `.lic.json`
- 呼叫 `verify` 邏輯（可直接搬 `license_program.js` 內函式進主程序）
- 驗證成功才啟用功能
- 將 `payload.permissions` 映射到 UI 和功能開關

## 6. 實務建議（重要）

- 私鑰必須離線保存，不要放進專案或客戶端
- 建議使用 `kid` 做金鑰輪替
- 防回滾是「提高成本」而非絕對防破解；若要更強，建議：
  - Windows DPAPI 保護狀態檔
  - 狀態寫入雙位置（APPDATA + PROGRAMDATA）交叉比對
  - 定期上線同步 server-side nonce（有網路時）
