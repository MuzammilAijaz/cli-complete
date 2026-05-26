# Goal
Implement a quick terminal based "tool" that can essentially turn natural language to linux commands.

## Workflow
- start program
- enter prompt "show all usb devices"
- model suggests options to execute:
  - `lsusb`
  - `usb-devices`
- user selects command to execute
- tool runs command and prints output

There might also be slightly more advanced ones like:
- "show all listening ports"
- model suggests:
  - `ss -tulpn`
  - `netstat -tulpn`

## The goal and scope of this project
- Focus on the exploring the idea itself and a minimal viable prototype (the cli tool).
- No complex parsing, sandboxing, or full shell agent behavior.
