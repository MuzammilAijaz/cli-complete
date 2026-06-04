# Overview
A quick terminal-based proof-of-concept tool that can essentially turn natural language to linux commands, all done locally.

> _WARNING:_ This project executes AI-generated shell commands and includes no safety mechanisms. It may suggest or run destructive, incorrect, or unintended commands. Use only in controlled environments and at your own risk. This project is intended solely as a proof of concept.

## Workflow
- start program
- enter prompt "show all usb devices"
- model suggests options to execute:
  - `lsusb`
  - `usb-devices`
- user selects command to execute
- tool runs command and prints output

## Goal and scope of this tool
- NOT intended for real-world use, as no safety features are implemented (e.g., protection against commands such as rm -rf).
- Focus on the exploring the idea itself and a minimal viable prototype (the cli tool and the application of NLP).
- No complex parsing, sandboxing, or full shell agent behavior.
- Cli implementation in python, so dont expect speed.
