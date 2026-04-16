const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopExcel', {
  open: () => ipcRenderer.invoke('excel:open'),                 // 你原本的：開 dialog 選檔
  openDefault: () => ipcRenderer.invoke('excel:openDefault'),   // 新增：固定預設檔
  openPath: (p) => ipcRenderer.invoke('excel:openPath', p),     // 新增：指定路徑
  saveAs: (payload) => ipcRenderer.invoke('excel:saveAs', payload), // 新增：另存新檔
  save: (payload) => ipcRenderer.invoke('excel:save', payload), // 你原本的：存回同檔
});

contextBridge.exposeInMainWorld('desktopAuth', {
  login: (payload) => ipcRenderer.invoke('auth:login', payload),
});

contextBridge.exposeInMainWorld('desktopCustomize', {
  listPlatforms: () => ipcRenderer.invoke('customize:listPlatforms'),
  savePlatform: (payload) => ipcRenderer.invoke('customize:savePlatform', payload),
  savePlatformAs: (payload) => ipcRenderer.invoke('customize:savePlatformAs', payload),
  loadPlatform: (payload) => ipcRenderer.invoke('customize:loadPlatform', payload),
});