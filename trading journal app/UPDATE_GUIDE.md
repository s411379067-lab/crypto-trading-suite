# 更新和發布流程

## 快速參考

### 開發階段
```bash
npm start  # 本地測試
```

### 發布新版本
```bash
./build-and-release.ps1  # 使用自動化腳本
```

或手動執行：
```bash
# 1. 更新 package.json 版本號
# 2. 提交到 Git
git add .
git commit -m "描述你的更新"
git push

# 3. 打包
npm run build:win

# 4. 推送到 GitHub Releases
```

## 版本號規則

- `patch`: 1.0.0 → 1.0.1 (修復 bug)
- `minor`: 1.0.0 → 1.1.0 (新功能)  
- `major`: 1.0.0 → 2.0.0 (大改版)

## 文件清單

每次打包前確保有這些文件：
- ✅ index.html (主介面)
- ✅ main.js (Electron 主進程)
- ✅ preload.js (安全上下文)
- ✅ js/ (儀表板模組)
- ✅ package.json (依賴配置)
- ✅ node_modules/ (依賴模組)

## 發布到 GitHub Releases

1. 標記版本：`git tag v1.1.0 && git push origin v1.1.0`
2. 前往：https://github.com/s411379067-lab/trade-journal-desktop/releases
3. 點擊 "Create a new release"
4. 選擇 tag v1.1.0
5. 上傳 `release\Trading Journal Setup 1.1.0.exe`
6. 發布

這樣其他使用者就能在 Releases 頁面下載你的最新版本！
