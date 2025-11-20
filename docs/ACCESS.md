# AWS / SSH Access Notes

- Key path (local): `~/.ssh/kalshi-key.pem` (permissions 600)
- SSH config entry: `~/.ssh/config` host `kalshi-aws`
  - HostName: `98.93.78.177`
  - User: `ec2-user` (switch to `ubuntu` if AMI is Ubuntu)
  - IdentityFile: `~/.ssh/kalshi-key.pem`
  - StrictHostKeyChecking: `accept-new`
- Quick connect: `ssh kalshi-aws`
- If you rotate the key, update both the file at `~/.ssh/kalshi-key.pem` and the `IdentityFile` path in `~/.ssh/config`.
