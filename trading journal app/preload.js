const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopExcel', {
  open: () => ipcRenderer.invoke('excel:open'),                 // 你原本的：開 dialog 選檔
  openDefault: () => ipcRenderer.invoke('excel:openDefault'),   // 新增：固定預設檔
  openPath: (p) => ipcRenderer.invoke('excel:openPath', p),     // 新增：指定路徑
  save: (payload) => ipcRenderer.invoke('excel:save', payload), // 你原本的：存回同檔
});