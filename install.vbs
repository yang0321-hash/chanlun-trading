Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Check if Python exists
Set objExec = objShell.Exec("python --version 2>&1")
strOutput = objExec.StdOut.ReadAll & objExec.StdErr.ReadAll

If InStr(strOutput, "Python") = 0 Then
    MsgBox "Python not found!" & vbCrLf & vbCrLf & _
           "Please install Python 3.9+ from:" & vbCrLf & _
           "https://www.python.org/downloads/" & vbCrLf & vbCrLf & _
           "Make sure to check 'Add Python to PATH' during installation.", _
           vbCritical, "Error"
    WScript.Quit 1
End If

' Show installation message
MsgBox "Python found: " & Trim(strOutput) & vbCrLf & vbCrLf & _
       "Click OK to install dependencies. This may take a few minutes...", _
       vbInformation, "Installing"

' Run pip install
objShell.Run "cmd /k python -m pip install akshare pandas numpy plotly loguru python-dotenv scipy matplotlib", 1, True

MsgBox "Installation complete! Now run run.bat", vbInformation, "Done"
