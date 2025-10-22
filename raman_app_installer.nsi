; NSIS Installer Script for Raman Spectroscopy Analysis Application
; Generated: October 21, 2025
; 
; This script creates a Windows installer for the Raman Spectroscopy application
; Requirements: NSIS 3.0+
; 
; To create installer, run:
;   makensis raman_app_installer.nsi
; 
; Output: raman_app_installer.exe

; ============== INCLUDES ==============
!include "MUI2.nsh"
!include "x64.nsh"

; ============== CONFIGURATION ==============

; Application information
!define APP_NAME "Raman Spectroscopy Analysis"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Raman Analysis Team"
!define APP_WEBSITE "https://github.com"
!define APP_EXE "raman_app.exe"

; Installation paths
!define INSTALL_DIR "$PROGRAMFILES\RamanApp"
!define UNINSTALL_EXE "uninstall.exe"

; Size estimate (approximate)
!define APP_ESTIMATED_SIZE 100000  ; ~100 MB in KB

; ============== MUI SETTINGS ==============

; Use Modern UI 2
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Language
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Japanese"

; ============== INSTALLER ATTRIBUTES ==============

Name "${APP_NAME} ${APP_VERSION}"
OutFile "raman_app_installer.exe"
InstallDir "${INSTALL_DIR}"
RequestExecutionLevel admin

; Show installation details
ShowInstDetails show
ShowUninstDetails show

; ============== SECTIONS ==============

Section "Install"
  SetOutPath "$INSTDIR"
  
  ; Set compression
  SetCompress auto
  
  ; Copy application files from staging directory
  ; Adjust path based on your build output structure
  File /r "dist_installer\raman_app_installer_staging\*.*"
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\${UNINSTALL_EXE}"
  
  ; Create Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\${UNINSTALL_EXE}"
  
  ; Create desktop shortcut (optional)
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
  
  ; Write registry entries for uninstall
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "DisplayName" "${APP_NAME} ${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "URLInfoAbout" "${APP_WEBSITE}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "DisplayIcon" "$INSTDIR\${APP_EXE}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "UninstallString" "$INSTDIR\${UNINSTALL_EXE}"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "EstimatedSize" "${APP_ESTIMATED_SIZE}"
  
  DetailPrint "Installation complete!"
SectionEnd

Section "Uninstall"
  ; Remove Start Menu shortcuts
  RMDir /r "$SMPROGRAMS\${APP_NAME}"
  
  ; Remove desktop shortcut
  Delete "$DESKTOP\${APP_NAME}.lnk"
  
  ; Remove application directory
  RMDir /r "$INSTDIR"
  
  ; Remove registry entries
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
  
  DetailPrint "Uninstallation complete!"
SectionEnd

; ============== HELPER FUNCTIONS ==============

Function .onInit
  ; Check for admin privileges (Windows Vista+)
  ${If} ${AtLeastWinVista}
    UserInfo::GetAccountType
    Pop $0
    ${If} $0 != "admin"
      MessageBox MB_OK "Administrator privileges required to install!"
      Quit
    ${EndIf}
  ${EndIf}
FunctionEnd

; ============== TRANSLATIONS ==============

; English
LangString DESC_Install ${LANG_ENGLISH} \
  "Install ${APP_NAME} ${APP_VERSION}. This will copy all necessary files and create shortcuts."

; Japanese
LangString DESC_Install ${LANG_JAPANESE} \
  "${APP_NAME} ${APP_VERSION} をインストールしています。必要なファイルをコピーしてショートカットを作成します。"

; ============== NOTES ==============
; 
; To customize this installer:
; 
; 1. Update APP_NAME, APP_VERSION, APP_PUBLISHER, etc. at the top
; 2. Ensure File /r path matches your PyInstaller output directory
; 3. Add icon: Change "icon" parameter in main section
; 4. Modify directory: Change INSTALL_DIR variable
; 5. Add license/readme: Uncomment MUI pages as needed
; 
; Common modifications:
; - Add license page: !insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
; - Add custom page: Page custom customPage
; - Change colors: Set MUI_BGCOLOR
; - Use 64-bit: ${If} ${RunningX64}
; 
; For more info: https://nsis.sourceforge.io/Docs/
