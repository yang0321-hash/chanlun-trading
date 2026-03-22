Set objShell = CreateObject("WScript.Shell")

' Change to script directory
objShell.CurrentDirectory = objShell.ParentFolderName

' Run the Python script
objShell.Run "cmd /k python examples\basic_usage.py", 1, True
