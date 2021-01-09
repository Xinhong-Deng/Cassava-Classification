VSCODE CONFIGURATION

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "starter_issam",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/trainval.py",
            "console": "integratedTerminal",
            "args":[
                "-e", "starter_issam",
                "-r", "1",
                "-d","/mnt/public/datasets/cassava",
                "-sb", "/mnt/public/results/debug/cassava",
                "-nw","0",
                "-v", "results/results.ipynb"
        ],
        },
    ]
}
```